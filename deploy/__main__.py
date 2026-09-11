"""The SQLForge serving stack on GKE: cluster, node pools, and workloads.

    pulumi up        brings the whole stack up from nothing
    pulumi destroy   takes it back down

Shape of it:

    Internet (optional LB) -> gateway Deployment (CPU pool, HPA)
                                   |
                              vllm Service
                                   |
                              vllm Deployment (GPU pool, 0->1 nodes)
                                   |
                              GCS: the GPTQ artifact, pulled by an init container

Two deliberate omissions from the stack: the GCS bucket holding the model and
the Artifact Registry repository holding the gateway image are created outside
this program (see PLAN.md). Both outlive any cluster, and `pulumi destroy`
should never be able to delete model weights or the image it deployed.
"""

import pulumi
import pulumi_gcp as gcp
import pulumi_kubernetes as k8s

config = pulumi.Config()
gcp_config = pulumi.Config("gcp")

PROJECT = gcp_config.require("project")
ZONE = gcp_config.require("zone")
REGION = gcp_config.require("region")

MODEL_BUCKET = config.require("modelBucket")
MODEL_PREFIX = config.require("modelPrefix")
GATEWAY_IMAGE = config.require("gatewayImage")
GPU_MAX_NODES = config.require_int("gpuMaxNodes")
VLLM_REPLICAS = config.require_int("vllmReplicas")
GATEWAY_MIN_REPLICAS = config.require_int("gatewayMinReplicas")
GATEWAY_MAX_REPLICAS = config.require_int("gatewayMaxReplicas")

# Public exposure is opt-in: with expose=false the gateway stays a ClusterIP
# and the demo is reachable only through `kubectl port-forward`.
EXPOSE_PUBLICLY = config.get_bool("expose") or False
# Secrets, so they are encrypted in the stack state rather than sitting in a
# config file. The Anthropic key is optional - without it the page serves the
# local model alone and says the comparison is switched off.
DEMO_TOKEN = config.require_secret("demoToken")
SERVICE_TOKEN = config.require_secret("serviceToken")
ANTHROPIC_API_KEY = config.get_secret("anthropicApiKey")

# The model is served under this name; clients and the gateway both use it.
SERVED_MODEL_NAME = "sqlforge-3b"
VLLM_IMAGE = "vllm/vllm-openai:v0.28.0"
NAMESPACE = "sqlforge"

# L4 capacity runs out in individual zones (us-central1-a was in STOCKOUT
# during M4), so the GPU pool may place its node in any zone of the region.
# GPU quota is regional, so this costs nothing in quota terms.
GPU_ZONES = [f"{REGION}-a", f"{REGION}-b", f"{REGION}-c"]


# --- Cluster -----------------------------------------------------------------

# Zonal control plane: GKE's free tier covers one zonal cluster's management
# fee, and a regional control plane buys HA we do not need for a single-GPU
# service.
cluster = gcp.container.Cluster(
    "sqlforge",
    name="sqlforge",
    location=ZONE,
    # The default pool is replaced immediately by the two pools below, but GKE
    # requires a node count at creation time.
    initial_node_count=1,
    remove_default_node_pool=True,
    deletion_protection=False,
    networking_mode="VPC_NATIVE",
    ip_allocation_policy=gcp.container.ClusterIpAllocationPolicyArgs(),
    release_channel=gcp.container.ClusterReleaseChannelArgs(channel="REGULAR"),
    # Workload Identity lets the vLLM pod read the model from GCS with no
    # service-account key anywhere in the cluster.
    workload_identity_config=gcp.container.ClusterWorkloadIdentityConfigArgs(
        workload_pool=f"{PROJECT}.svc.id.goog"
    ),
    # Managed Prometheus scrapes the gateway's and vLLM's /metrics without a
    # Prometheus server to operate.
    monitoring_config=gcp.container.ClusterMonitoringConfigArgs(
        managed_prometheus=gcp.container.ClusterMonitoringConfigManagedPrometheusArgs(
            enabled=True
        ),
        enable_components=["SYSTEM_COMPONENTS"],
    ),
)

# Everything that is not model inference: the gateway, and GKE's own system
# pods. Small and always on.
system_pool = gcp.container.NodePool(
    "system",
    name="system",
    cluster=cluster.name,
    location=ZONE,
    autoscaling=gcp.container.NodePoolAutoscalingArgs(min_node_count=1, max_node_count=3),
    management=gcp.container.NodePoolManagementArgs(auto_repair=True, auto_upgrade=True),
    node_config=gcp.container.NodePoolNodeConfigArgs(
        # 4 vCPU, not 2: the gateway tier plus the benchmark driver on a
        # 2-vCPU node contend for the same cores, and measured throughput
        # collapses above ~64 concurrent clients for reasons that have
        # nothing to do with the model or the GPU.
        machine_type="e2-standard-4",
        disk_size_gb=50,
        disk_type="pd-balanced",
        oauth_scopes=["https://www.googleapis.com/auth/cloud-platform"],
        workload_metadata_config=gcp.container.NodePoolNodeConfigWorkloadMetadataConfigArgs(
            mode="GKE_METADATA"
        ),
        labels={"workload": "system"},
    ),
)

# The GPU pool scales to zero. A pending vLLM pod brings a node up (~4 min,
# most of it the driver install and the image pull); deleting the deployment
# takes it back down, and an idle cluster then costs only the system pool.
gpu_pool = gcp.container.NodePool(
    "gpu",
    name="gpu",
    cluster=cluster.name,
    location=ZONE,
    node_locations=GPU_ZONES,
    autoscaling=gcp.container.NodePoolAutoscalingArgs(
        min_node_count=0, max_node_count=GPU_MAX_NODES
    ),
    management=gcp.container.NodePoolManagementArgs(auto_repair=True, auto_upgrade=True),
    node_config=gcp.container.NodePoolNodeConfigArgs(
        machine_type="g2-standard-8",  # 8 vCPU, 32GB, 1x NVIDIA L4
        disk_size_gb=100,
        disk_type="pd-balanced",
        oauth_scopes=["https://www.googleapis.com/auth/cloud-platform"],
        workload_metadata_config=gcp.container.NodePoolNodeConfigWorkloadMetadataConfigArgs(
            mode="GKE_METADATA"
        ),
        labels={"workload": "vllm"},
        guest_accelerators=[
            gcp.container.NodePoolNodeConfigGuestAcceleratorArgs(
                type="nvidia-l4",
                count=1,
                # GKE installs and manages the driver; without this the node
                # comes up with no usable GPU.
                gpu_driver_installation_config=gcp.container.NodePoolNodeConfigGuestAcceleratorGpuDriverInstallationConfigArgs(
                    gpu_driver_version="LATEST"
                ),
            )
        ],
        # GKE taints GPU nodes itself; declaring the taint keeps Pulumi from
        # seeing a permanent diff against the live node pool.
        taints=[
            gcp.container.NodePoolNodeConfigTaintArgs(
                key="nvidia.com/gpu", value="present", effect="NO_SCHEDULE"
            )
        ],
    ),
)


# --- Identity: let the vLLM pod read the model from GCS ----------------------

model_reader = gcp.serviceaccount.Account(
    "model-reader",
    account_id="sqlforge-model-reader",
    display_name="SQLForge: reads the serving artifact from GCS",
)

gcp.storage.BucketIAMMember(
    "model-reader-object-viewer",
    bucket=MODEL_BUCKET,
    role="roles/storage.objectViewer",
    member=model_reader.email.apply(lambda email: f"serviceAccount:{email}"),
)

# Binds the Kubernetes service account below to the Google one.
#
# depends_on the cluster, and not for tidiness: the member references the
# project's Workload Identity pool (PROJECT.svc.id.goog), which does not exist
# until a cluster with Workload Identity has been created. Without this,
# Pulumi creates the binding in parallel with the cluster and IAM rejects it
# with "Identity Pool does not exist" - and since Pulumi infers dependencies
# from data flow, a member string built from a plain f-string carries no
# reference to the cluster for it to infer.
gcp.serviceaccount.IAMMember(
    "model-reader-workload-identity",
    service_account_id=model_reader.name,
    role="roles/iam.workloadIdentityUser",
    member=f"serviceAccount:{PROJECT}.svc.id.goog[{NAMESPACE}/vllm]",
    opts=pulumi.ResourceOptions(depends_on=[cluster]),
)


# --- Kubernetes provider -----------------------------------------------------

# The kubeconfig carries a short-lived OAuth token from the credentials this
# program already runs with, rather than an exec block calling
# gke-gcloud-auth-plugin. One less binary to have installed, and it works the
# same in CI; the token is refreshed on every run.
client_config = gcp.organizations.get_client_config()


def _ca_certificate(master_auth) -> str:
    """Read the cluster CA out of master_auth however the engine hands it over.

    gcp.container.ClusterMasterAuth subclasses dict and keys it in snake_case,
    but inside an apply the engine can pass a plain dict instead of the typed
    class, where attribute access fails - so read it as a mapping and accept
    either casing. Worth noting this only surfaces once the cluster exists:
    while it is being created the value is unknown, Pulumi skips the apply
    body entirely, and `pulumi preview` reports no problem.
    """
    return master_auth.get("cluster_ca_certificate") or master_auth["clusterCaCertificate"]


kubeconfig = pulumi.Output.all(cluster.name, cluster.endpoint, cluster.master_auth).apply(
    lambda args: f"""apiVersion: v1
kind: Config
clusters:
- cluster:
    server: https://{args[1]}
    certificate-authority-data: {_ca_certificate(args[2])}
  name: {args[0]}
contexts:
- context:
    cluster: {args[0]}
    user: {args[0]}
  name: {args[0]}
current-context: {args[0]}
users:
- name: {args[0]}
  user:
    token: {client_config.access_token}
"""
)

k8s_provider = k8s.Provider(
    "gke",
    kubeconfig=kubeconfig,
    # Nodes must exist before workloads are applied, or the first apply races
    # the cluster's own readiness.
    opts=pulumi.ResourceOptions(depends_on=[system_pool, gpu_pool]),
)
k8s_opts = pulumi.ResourceOptions(provider=k8s_provider)

namespace = k8s.core.v1.Namespace(
    "sqlforge",
    metadata={"name": NAMESPACE},
    opts=k8s_opts,
)
ns_opts = pulumi.ResourceOptions(provider=k8s_provider, depends_on=[namespace])

vllm_sa = k8s.core.v1.ServiceAccount(
    "vllm",
    metadata={
        "name": "vllm",
        "namespace": NAMESPACE,
        "annotations": {"iam.gke.io/gcp-service-account": model_reader.email},
    },
    opts=ns_opts,
)


# --- vLLM --------------------------------------------------------------------

vllm_labels = {"app": "vllm"}

vllm = k8s.apps.v1.Deployment(
    "vllm",
    metadata={"name": "vllm", "namespace": NAMESPACE},
    spec={
        "replicas": VLLM_REPLICAS,
        "selector": {"match_labels": vllm_labels},
        # One GPU, one pod: a rolling update would need a second GPU to bring
        # up the new pod first, so replace instead of surging.
        "strategy": {"type": "Recreate"},
        "template": {
            "metadata": {"labels": vllm_labels},
            "spec": {
                "service_account_name": "vllm",
                # Kubernetes otherwise injects a Docker-link-style env var per
                # Service in the namespace - and the Service in front of this
                # pod is named "vllm", which yields
                # VLLM_PORT=tcp://<clusterIP>:8000. That collides with vLLM's
                # own VLLM_PORT setting, and the server exits: "VLLM_PORT
                # 'tcp://...' appears to be a URI". Service links are legacy
                # compatibility nobody here uses, so turn them off rather than
                # rename the Service and lose the obvious DNS name.
                "enable_service_links": False,
                "node_selector": {"cloud.google.com/gke-accelerator": "nvidia-l4"},
                "tolerations": [
                    {
                        "key": "nvidia.com/gpu",
                        "operator": "Equal",
                        "value": "present",
                        "effect": "NoSchedule",
                    }
                ],
                # The artifact is pulled at pod start rather than baked into an
                # image: the model is versioned in GCS, so swapping it is a
                # config change, not a multi-gigabyte image rebuild.
                "init_containers": [
                    {
                        "name": "fetch-model",
                        "image": "google/cloud-sdk:slim",
                        "command": ["/bin/sh", "-c"],
                        "args": [
                            (
                                f"gcloud storage rsync -r gs://{MODEL_BUCKET}/{MODEL_PREFIX}"
                                " /model && ls -la /model"
                            )
                        ],
                        "volume_mounts": [{"name": "model", "mount_path": "/model"}],
                        "resources": {
                            "requests": {"cpu": "500m", "memory": "512Mi"},
                            "limits": {"cpu": "2", "memory": "2Gi"},
                        },
                    }
                ],
                "containers": [
                    {
                        "name": "vllm",
                        "image": VLLM_IMAGE,
                        "args": [
                            "--model",
                            "/model",
                            "--served-model-name",
                            SERVED_MODEL_NAME,
                            "--max-model-len",
                            "4096",
                            "--gpu-memory-utilization",
                            "0.90",
                            "--port",
                            "8000",
                        ],
                        "ports": [{"name": "http", "container_port": 8000}],
                        "env": [
                            # The image defaults to writing caches under a
                            # read-only home; keep them on the pod's own disk.
                            {"name": "HF_HOME", "value": "/tmp/hf"},
                            {"name": "VLLM_CACHE_ROOT", "value": "/tmp/vllm"},
                        ],
                        "resources": {
                            "requests": {"cpu": "4", "memory": "16Gi", "nvidia.com/gpu": "1"},
                            "limits": {"memory": "24Gi", "nvidia.com/gpu": "1"},
                        },
                        "volume_mounts": [
                            {"name": "model", "mount_path": "/model", "read_only": True},
                            {"name": "cache", "mount_path": "/tmp"},
                            # vLLM's workers communicate through shared memory;
                            # the 64MB default /dev/shm is not enough.
                            {"name": "shm", "mount_path": "/dev/shm"},
                        ],
                        # Weights load in ~90s and the artifact download runs
                        # before that, so readiness gets a long startup budget
                        # and a tight steady-state check.
                        "startup_probe": {
                            "http_get": {"path": "/health", "port": "http"},
                            "period_seconds": 10,
                            "failure_threshold": 60,
                        },
                        "readiness_probe": {
                            "http_get": {"path": "/health", "port": "http"},
                            "period_seconds": 5,
                            "failure_threshold": 3,
                        },
                        # Liveness is deliberately slack: a busy engine under
                        # load must not be mistaken for a hung one.
                        "liveness_probe": {
                            "http_get": {"path": "/health", "port": "http"},
                            "period_seconds": 30,
                            "timeout_seconds": 10,
                            "failure_threshold": 5,
                        },
                    }
                ],
                "volumes": [
                    {"name": "model", "empty_dir": {"size_limit": "10Gi"}},
                    {"name": "cache", "empty_dir": {"size_limit": "10Gi"}},
                    {"name": "shm", "empty_dir": {"medium": "Memory", "size_limit": "2Gi"}},
                ],
            },
        },
    },
    opts=pulumi.ResourceOptions(
        provider=k8s_provider,
        depends_on=[namespace, vllm_sa],
        # A cold start is node provisioning + driver install + a 6GB image
        # pull + a 2.1GB model download + weight load, which comfortably
        # exceeds the default 10-minute await.
        custom_timeouts=pulumi.CustomTimeouts(create="30m", update="30m", delete="15m"),
    ),
)

vllm_service = k8s.core.v1.Service(
    "vllm",
    metadata={"name": "vllm", "namespace": NAMESPACE},
    spec={
        "selector": vllm_labels,
        "ports": [{"name": "http", "port": 8000, "target_port": "http"}],
    },
    opts=ns_opts,
)


# --- Gateway -----------------------------------------------------------------

gateway_labels = {"app": "gateway"}

# One Secret for everything the gateway must not carry in its manifest. The
# Anthropic key is written as an empty string when unset so the container's
# env wiring does not have to branch.
gateway_secret = k8s.core.v1.Secret(
    "gateway",
    metadata={"name": "gateway", "namespace": NAMESPACE},
    string_data={
        "demo-token": DEMO_TOKEN,
        "service-token": SERVICE_TOKEN,
        "anthropic-api-key": ANTHROPIC_API_KEY if ANTHROPIC_API_KEY is not None else "",
    },
    opts=ns_opts,
)

gateway = k8s.apps.v1.Deployment(
    "gateway",
    metadata={"name": "gateway", "namespace": NAMESPACE},
    spec={
        "replicas": GATEWAY_MIN_REPLICAS,
        "selector": {"match_labels": gateway_labels},
        "template": {
            "metadata": {"labels": gateway_labels},
            "spec": {
                # Same reasoning as the vLLM pod: no legacy service-link env
                # vars, so no Service name can collide with app config.
                "enable_service_links": False,
                "node_selector": {"workload": "system"},
                # Spread replicas across nodes so a node scale-down cannot
                # take out the whole tier.
                "topology_spread_constraints": [
                    {
                        "max_skew": 1,
                        "topology_key": "kubernetes.io/hostname",
                        "when_unsatisfiable": "ScheduleAnyway",
                        "label_selector": {"match_labels": gateway_labels},
                    }
                ],
                "containers": [
                    {
                        "name": "gateway",
                        "image": GATEWAY_IMAGE,
                        "ports": [{"name": "http", "container_port": 8080}],
                        "env": [
                            {
                                "name": "SQLFORGE_UPSTREAM",
                                "value": f"http://vllm.{NAMESPACE}.svc.cluster.local:8000/v1",
                            },
                            {"name": "SQLFORGE_MODEL", "value": SERVED_MODEL_NAME},
                            {"name": "SQLFORGE_TIMEOUT_S", "value": "30"},
                            # Public rate limits, applied per replica (see
                            # security.py): a demo visitor gets a handful of
                            # queries a minute, and the paid comparison gets
                            # a much smaller allowance because every call
                            # costs money.
                            {"name": "SQLFORGE_RATE_PER_MINUTE", "value": "12"},
                            {"name": "SQLFORGE_RATE_BURST", "value": "6"},
                            {"name": "SQLFORGE_DAILY_LIMIT", "value": "2000"},
                            {"name": "SQLFORGE_PAID_RATE_PER_MINUTE", "value": "4"},
                            {"name": "SQLFORGE_PAID_BURST", "value": "2"},
                            {"name": "SQLFORGE_PAID_DAILY_LIMIT", "value": "200"},
                            {
                                "name": "SQLFORGE_DEMO_TOKEN",
                                "value_from": {
                                    "secret_key_ref": {"name": "gateway", "key": "demo-token"}
                                },
                            },
                            {
                                "name": "SQLFORGE_SERVICE_TOKEN",
                                "value_from": {
                                    "secret_key_ref": {"name": "gateway", "key": "service-token"}
                                },
                            },
                            {
                                "name": "ANTHROPIC_API_KEY",
                                "value_from": {
                                    "secret_key_ref": {
                                        "name": "gateway",
                                        "key": "anthropic-api-key",
                                    }
                                },
                            },
                        ],
                        # The CPU request is what the HPA measures against, and
                        # the guardrail's sqlglot parse is the real CPU cost
                        # per request.
                        "resources": {
                            "requests": {"cpu": "200m", "memory": "192Mi"},
                            "limits": {"cpu": "1", "memory": "512Mi"},
                        },
                        # Liveness must not depend on vLLM: restarting the
                        # gateway cannot fix a model server that is down.
                        "liveness_probe": {
                            "http_get": {"path": "/healthz", "port": "http"},
                            "period_seconds": 10,
                            "failure_threshold": 3,
                        },
                        # Readiness does, so a gateway with no model behind it
                        # is pulled out of the Service instead of serving 502s.
                        "readiness_probe": {
                            "http_get": {"path": "/readyz", "port": "http"},
                            "period_seconds": 10,
                            "failure_threshold": 3,
                        },
                        "security_context": {
                            "run_as_non_root": True,
                            "run_as_user": 1001,
                            "allow_privilege_escalation": False,
                            "read_only_root_filesystem": True,
                            "capabilities": {"drop": ["ALL"]},
                        },
                    }
                ],
            },
        },
    },
    opts=pulumi.ResourceOptions(
        provider=k8s_provider,
        # Ordered after vLLM because the gateway's readiness probe checks the
        # model server: deploying it first would just mean a tier of NotReady
        # pods while the GPU node comes up.
        depends_on=[namespace, vllm_service, vllm, gateway_secret],
        custom_timeouts=pulumi.CustomTimeouts(create="15m", update="15m"),
    ),
)

# A network load balancer when exposed: the demo token and the rate limits are
# what protect the GPU, so the only thing this adds is reachability.
# externalTrafficPolicy=Local preserves the client IP, which the per-IP rate
# limiter needs to mean anything.
gateway_service = k8s.core.v1.Service(
    "gateway",
    metadata={"name": "gateway", "namespace": NAMESPACE},
    spec={
        "type": "LoadBalancer" if EXPOSE_PUBLICLY else "ClusterIP",
        **({"external_traffic_policy": "Local"} if EXPOSE_PUBLICLY else {}),
        "selector": gateway_labels,
        # Port 80 in both modes so the public URL needs no port suffix and
        # in-cluster callers use one address either way. The container still
        # listens on 8080; targetPort bridges them.
        "ports": [{"name": "http", "port": 80, "target_port": "http"}],
    },
    opts=pulumi.ResourceOptions(
        provider=k8s_provider,
        depends_on=[namespace],
        # Kubernetes merges port lists by port number, so editing the port
        # adds a second entry instead of replacing the first and the Service
        # is rejected for a duplicate port name. Replacing sidesteps the
        # merge; a Service is cheap to recreate (the load balancer IP is
        # reallocated, which is why the URL is an output rather than a
        # promise).
        replace_on_changes=["spec.ports"],
        delete_before_replace=True,
    ),
)

# The gateway tier is the only thing worth horizontally scaling here: vLLM is
# pinned to the single GPU the project's quota allows, so an HPA on queue
# depth would sit at "desired 2, current 1" forever. CPU is the honest signal
# for this tier - per request the gateway renders a prompt, parses the
# completion with sqlglot, and checks the guardrail, and that is CPU work.
gateway_hpa = k8s.autoscaling.v2.HorizontalPodAutoscaler(
    "gateway",
    metadata={"name": "gateway", "namespace": NAMESPACE},
    spec={
        "scale_target_ref": {"api_version": "apps/v1", "kind": "Deployment", "name": "gateway"},
        "min_replicas": GATEWAY_MIN_REPLICAS,
        "max_replicas": GATEWAY_MAX_REPLICAS,
        "metrics": [
            {
                "type": "Resource",
                "resource": {
                    "name": "cpu",
                    "target": {"type": "Utilization", "average_utilization": 70},
                },
            }
        ],
        "behavior": {
            # Scale up promptly, shed slowly: a load test's ramp should add
            # replicas, and its end should not immediately remove them.
            "scale_up": {"stabilization_window_seconds": 30},
            "scale_down": {"stabilization_window_seconds": 300},
        },
    },
    opts=pulumi.ResourceOptions(provider=k8s_provider, depends_on=[gateway]),
)


# --- Observability -----------------------------------------------------------

# Managed Prometheus scrapes both tiers. vLLM's queue-depth series
# (vllm:num_requests_waiting) is the signal a GPU-tier HPA would use once the
# GPU quota allows more than one replica.
for name, app, port in (("gateway", "gateway", "http"), ("vllm", "vllm", "http")):
    k8s.apiextensions.CustomResource(
        f"podmonitoring-{name}",
        api_version="monitoring.googleapis.com/v1",
        kind="PodMonitoring",
        metadata={"name": name, "namespace": NAMESPACE},
        spec={
            "selector": {"matchLabels": {"app": app}},
            "endpoints": [{"port": port, "interval": "15s", "path": "/metrics"}],
        },
        opts=ns_opts,
    )


# Secret so the token does not land in plaintext in the state file or in
# terminal output; read it with:
#   pulumi stack output kubeconfig --show-secrets > /tmp/sqlforge.kubeconfig
pulumi.export("kubeconfig", pulumi.Output.secret(kubeconfig))
pulumi.export("cluster_name", cluster.name)
pulumi.export("cluster_zone", cluster.location)
pulumi.export("gpu_pool_max_nodes", pulumi.Output.from_input(GPU_MAX_NODES))
pulumi.export("gateway_service", gateway_service.metadata["name"])
if EXPOSE_PUBLICLY:
    pulumi.export(
        "demo_url",
        gateway_service.status.apply(
            lambda status: f"http://{status['load_balancer']['ingress'][0]['ip']}"
            if status and status.get("load_balancer", {}).get("ingress")
            else "pending"
        ),
    )
