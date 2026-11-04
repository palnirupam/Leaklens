// Run with Playwright browser_run_code_unsafe(filename=...) and matching nonempty API_KEY / VITE_API_KEY.
async (page) => {
  const assert = (condition, message) => { if (!condition) throw new Error(message); };
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("http://127.0.0.1:5173");
  await page.getByText("Backend online").waitFor();
  await page.locator("nav").getByRole("button", { name: "New Investigation" }).click();

  const csv = "x,late_feature,event_time,customer_id,target\n" + Array.from({ length: 160 }, (_, i) =>
    `${Math.sin(i * 13.37)},${i % 2},${new Date(Date.UTC(2024, 0, i + 1)).toISOString()},${i},${i % 2 ? "yes" : "no"}`
  ).join("\n");
  await page.locator('input[type="file"]').nth(0).setInputFiles({
    name: "browser_check.csv", mimeType: "text/csv", buffer: Buffer.from(csv),
  });
  await page.locator('input[type="file"]').nth(1).setInputFiles({
    name: "training.py", mimeType: "text/plain", buffer: Buffer.from(
      "from sklearn.preprocessing import StandardScaler\nfrom sklearn.model_selection import train_test_split\nX = StandardScaler().fit_transform(X)\nX_train, X_test = train_test_split(X)\n"
    ),
  });
  const target = page.getByPlaceholder("e.g. y", { exact: true });
  await target.click();
  await target.pressSequentially("target");
  assert(await target.inputValue() === "target", "Typing lost characters");
  assert(await target.evaluate((input) => document.activeElement === input), "Typing lost focus");
  await page.getByRole("spinbutton").fill("47");
  await page.getByPlaceholder("e.g. timestamp").fill("event_time");
  await page.getByPlaceholder("e.g. customer_id").fill("customer_id");
  await page.getByPlaceholder("e.g. duration, future_price").fill("late_feature");
  await page.getByRole("checkbox").check();
  await page.getByRole("checkbox").uncheck();
  const selectedFiles = await page.locator('input[type="file"]').evaluateAll((inputs) => inputs.map((input) => input.files.length));
  assert(selectedFiles.every((count) => count === 1), "Editing configuration cleared an upload");

  const createdResponse = page.waitForResponse((response) =>
    response.url().endsWith("/api/investigations") && response.request().method() === "POST"
  );
  await page.getByRole("button", { name: "Start Investigation", exact: true }).click();
  const response = await createdResponse;
  assert(response.status() === 202, "Investigation upload failed");
  assert(Boolean(await response.request().headerValue("x-api-key")), "Start dev servers with matching API_KEY and VITE_API_KEY");
  const investigation = await response.json();
  await page.locator(".meta-row .status-completed,.meta-row .status-failed").waitFor({ timeout: 30000 });
  assert(await page.locator(".meta-row .status-completed").count() === 1, "Investigation failed");
  await page.locator("nav").getByRole("button", { name: "Findings" }).click();
  await page.locator(".finding-card").first().waitFor();
  await page.locator("nav").getByRole("button", { name: "Reports" }).click();
  await page.getByText("Report summary", { exact: true }).waitFor();

  for (const format of ["json", "html"]) {
    const reportResponse = page.waitForResponse((r) => r.url().endsWith(`/report?format=${format}`));
    const downloaded = page.waitForEvent("download");
    await page.getByRole("button", { name: `Download ${format.toUpperCase()} report`, exact: false }).click();
    const report = await reportResponse;
    const download = await downloaded;
    assert(report.status() === 200, "Authenticated report request failed");
    assert(Boolean(await report.request().headerValue("x-api-key")), "Report omitted the API key");
    assert(download.suggestedFilename() === `investigation_${investigation.id}.${format}`, "Wrong download filename");
    assert(await download.failure() === null, "Browser download failed");
    const chunks = [];
    for await (const chunk of await download.createReadStream()) chunks.push(chunk);
    const body = Buffer.concat(chunks).toString("utf8");
    assert(format === "json" ? JSON.parse(body).id === investigation.id : body.includes("<html>"), "Invalid report contents");
  }

  const preview = page.getByRole("button", { name: "Quick preview", exact: false });
  await preview.focus();
  await preview.press("Enter");
  assert(await page.locator("#report-preview").isVisible(), "Preview is not keyboard accessible");

  const reportRoute = "**/api/investigations/*/report?format=json";
  await page.route(reportRoute, (route) => route.fulfill({
    status: 401, contentType: "application/json", body: '{"detail":"Invalid or missing API key"}',
  }));
  try {
    await page.getByRole("button", { name: "Download JSON report", exact: false }).click();
    await page.getByRole("alert").filter({ hasText: "401" }).waitFor();
  } finally {
    await page.unroute(reportRoute);
  }
  const retry = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download JSON report", exact: false }).click();
  await retry;
  assert(await page.getByRole("alert").count() === 0, "Download error did not clear on retry");
  assert(errors.length === 0, `Browser errors: ${errors.join(", ")}`);
  return { status: "passed", investigation: investigation.id, checks: ["upload retention", "input focus", "investigation", "authenticated JSON/HTML downloads", "keyboard preview", "download error/retry"] };
}
