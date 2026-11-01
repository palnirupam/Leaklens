def main() -> None:
    import sys, time, tempfile, os
    sys.path.insert(0, "backend")
    from fastapi.testclient import TestClient
    from app.data.synthetic import generate_case
    from app.main import app

    client = TestClient(app)

    # health
    h = client.get("/api/health")
    assert h.status_code == 200, h.text
    print("health:", h.json())

    # build a target-derived case
    case = generate_case("target_derived_feature", seed=11)
    with tempfile.TemporaryDirectory() as td:
        csv_path = os.path.join(td, "data.csv")
        case.df.to_csv(csv_path, index=False)
        with open(csv_path, "rb") as f:
            resp = client.post(
                "/api/investigations",
                data={"target_column": "target", "gemini_enabled": "false", "random_seed": "11"},
                files={"file": ("data.csv", f, "text/csv")},
            )
        print("create status:", resp.status_code, resp.json().get("id"), resp.json().get("status"))
        inv_id = resp.json()["id"]

        # poll
        for _ in range(120):
            r = client.get(f"/api/investigations/{inv_id}").json()
            if r["status"] in ("completed", "failed"):
                break
            time.sleep(0.5)
        print("final status:", r["status"], "stage:", r.get("stage"))
        print("findings:", r["finding_count"])
        print("experiments:", r["experiment_count"])

        findings = client.get(f"/api/investigations/{inv_id}/findings").json()
        for f in findings:
            print("  finding:", f["category"], f["evidence_status"], f["severity"])
        print("summary:", r.get("summary"))

        rep = client.get(f"/api/investigations/{inv_id}/report", params={"format": "json"})
        print("report json:", rep.status_code, "bytes", len(rep.content))
        rep_html = client.get(f"/api/investigations/{inv_id}/report", params={"format": "html"})
        print("report html:", rep_html.status_code, "bytes", len(rep_html.content))

        stats = client.get("/api/stats").json()
        print("stats total:", stats["total_investigations"], "confirmed:", stats["confirmed_findings"])

    print("API_SMOKE_OK")


if __name__ == "__main__":
    main()
