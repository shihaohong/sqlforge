/**
 * End-to-end test of the demo page against a running gateway.
 *
 *   npx playwright test                      # expects a gateway on :8080
 *   BASE_URL=http://localhost:8080 npx playwright test
 *
 * This is the only test that exercises the streaming path in a browser, which
 * is where SSE actually has to work. It also captures the screenshots used to
 * review the page's layout.
 */

import { expect, test } from "@playwright/test";

test("asks both engines and shows executed rows", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByRole("heading", { name: "SQLForge" })).toBeVisible();
  await expect(page.locator("select")).toBeVisible();

  // Take the first suggested question so the run has a gold query to compare
  // against, which is what exercises the correctness verdict.
  const suggestion = page.locator("button", { hasText: /\?$/ }).first();
  await suggestion.click();
  await page.screenshot({ path: "screenshots/01-idle.png", fullPage: true });

  await page.getByRole("button", { name: "Ask both" }).click();

  const localPanel = page.locator("section", { hasText: "Fine-tuned Llama-3.2 3B" });
  // Generated SQL should appear, and it should be a SELECT.
  await expect(localPanel.locator("pre")).toContainText(/select/i, { timeout: 60_000 });
  // Then the rows it returned, with the gold verdict beside them.
  await expect(localPanel.getByText(/rows (match|differ)|no gold query/)).toBeVisible({
    timeout: 60_000,
  });
  await expect(localPanel.getByText(/row(s)?$|rows shown/)).toBeVisible({ timeout: 60_000 });

  await page.waitForTimeout(1500); // let the Claude column settle too
  await page.screenshot({ path: "screenshots/02-answered.png", fullPage: true });
});

test("renders at a phone width", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "SQLForge" })).toBeVisible();
  await page.screenshot({ path: "screenshots/03-mobile.png", fullPage: true });
});
