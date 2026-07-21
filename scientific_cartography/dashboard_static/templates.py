"""HTML templates for static dashboard pages."""

import html


def _esc(value) -> str:
    """Escape a value for safe HTML interpolation (handles None and non-str)."""
    return html.escape("" if value is None else str(value), quote=True)


BASE_CSS = """
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; line-height: 1.6; color: #333; background: #f5f5f5; }
  .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
  header { background: white; padding: 20px; margin-bottom: 20px; border-bottom: 2px solid #ddd; }
  header h1 { margin-bottom: 10px; }
  header .meta { font-size: 0.9em; color: #666; }
  nav { background: white; padding: 0; margin-bottom: 20px; border-bottom: 1px solid #ddd; }
  nav ul { list-style: none; display: flex; flex-wrap: wrap; }
  nav li { margin: 0; }
  nav a { display: block; padding: 12px 16px; color: #0066cc; text-decoration: none; border-bottom: 3px solid transparent; }
  nav a:hover { background: #f0f0f0; }
  nav a.active { color: #0066cc; border-bottom-color: #0066cc; }
  main { background: white; padding: 20px; margin-bottom: 20px; }
  h2 { margin: 20px 0 10px 0; padding-bottom: 10px; border-bottom: 1px solid #ddd; }
  h3 { margin: 15px 0 5px 0; }
  table { width: 100%; border-collapse: collapse; margin: 10px 0; }
  th, td { padding: 8px; text-align: left; border-bottom: 1px solid #eee; }
  th { background: #f9f9f9; font-weight: bold; }
  tr:hover { background: #f9f9f9; }
  .warning { background: #fff3cd; border: 1px solid #ffc107; padding: 10px; margin: 10px 0; border-radius: 4px; }
  .error { background: #f8d7da; border: 1px solid #f5c6cb; padding: 10px; margin: 10px 0; border-radius: 4px; }
  .success { background: #d4edda; border: 1px solid #c3e6cb; padding: 10px; margin: 10px 0; border-radius: 4px; }
  .info { background: #d1ecf1; border: 1px solid #bee5eb; padding: 10px; margin: 10px 0; border-radius: 4px; }
  .governance { background: #f0f0f0; border: 1px solid #999; padding: 15px; margin: 10px 0; border-radius: 4px; }
  .governance-flag { display: flex; padding: 4px 0; }
  .governance-flag-name { width: 40%; font-weight: bold; }
  .governance-flag-value { width: 60%; }
  .flag-true { color: green; }
  .flag-false { color: red; }
  .flag-false.expected { color: green; }
  code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-family: 'Courier New', monospace; }
  footer { text-align: center; color: #999; font-size: 0.9em; padding: 20px; }
  .page-list { list-style: none; }
  .page-list li { padding: 4px 0; }
  .page-list a { color: #0066cc; text-decoration: none; }
  .page-list a:hover { text-decoration: underline; }
</style>
"""


def html_page(title: str, content: str, nav_items: list = None, current_page: str = "") -> str:
    """Generate a complete HTML page with header, nav, and content."""
    if nav_items is None:
        nav_items = []

    nav_html = ""
    if nav_items:
        nav_html = "<nav><ul>"
        for item_name, item_url in nav_items:
            is_active = item_name == current_page
            active_class = "active" if is_active else ""
            nav_html += f'<li><a href="{_esc(item_url)}" class="{active_class}">{_esc(item_name)}</a></li>'
        nav_html += "</ul></nav>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{_esc(title)}</title>
  {BASE_CSS}
</head>
<body>
  <header>
    <h1>Scientific Cartography LangGraph Dashboard</h1>
    <div class="meta">Read-only artifact browser for LG1/LG2/LG3 review outputs</div>
  </header>
  {nav_html}
  <div class="container">
    <main>
      {content}
    </main>
  </div>
  <footer>
    <p>Dashboard generated: <code>scientific_cartography/dashboard_static/generator.py</code></p>
    <p>Governance: READ_ONLY_DIAGNOSTIC | NO_SCORING | NO_AUTOMATION_APPROVAL</p>
  </footer>
</body>
</html>"""


def index_template(
    artifact_dir: str, as_of_date: str, pages: list, missing_artifacts: list, warnings: list, nav_items: list = None
) -> str:
    """Index page showing available pages and status."""
    warnings_html = ""
    if missing_artifacts:
        warnings_html += f'<div class="warning"><strong>Missing Artifacts:</strong> {", ".join(_esc(m) for m in missing_artifacts)}</div>'
    if warnings:
        for warning in warnings:
            warnings_html += f'<div class="warning">{_esc(warning)}</div>'

    pages_html = '<ul class="page-list">'
    for page_name, page_file, description in pages:
        pages_html += f'<li><a href="{_esc(page_file)}">{_esc(page_name)}</a> — {_esc(description)}</li>'
    pages_html += "</ul>"

    governance_info = """
    <div class="governance">
      <h3>Governance Status</h3>
      <p>This dashboard is a <strong>read-only artifact browser</strong>.</p>
      <div class="governance-flag">
        <div class="governance-flag-name">READ_ONLY_DIAGNOSTIC:</div>
        <div class="governance-flag-value flag-true">✓ true</div>
      </div>
      <div class="governance-flag">
        <div class="governance-flag-name">NO_SCORING:</div>
        <div class="governance-flag-value flag-true">✓ true</div>
      </div>
      <div class="governance-flag">
        <div class="governance-flag-name">AUTOMATION_APPROVAL:</div>
        <div class="governance-flag-value flag-false expected">✓ false</div>
      </div>
      <p style="margin-top: 10px; font-size: 0.9em; color: #666;">
        No scoring is performed. No investment recommendation is made. No production pipeline action is authorized.
      </p>
    </div>
    """

    content = f"""
    <h2>Dashboard Overview</h2>
    <p><strong>Artifact Directory:</strong> <code>{_esc(artifact_dir)}</code></p>
    <p><strong>As-of Date:</strong> <code>{_esc(as_of_date or "unknown")}</code></p>

    {warnings_html}

    <h2>Available Pages</h2>
    {pages_html}

    {governance_info}
    """

    return html_page("Dashboard Index", content, nav_items, "Index")


def review_runs_template(artifact_dir: str, review_data: dict, nav_items: list) -> str:
    """Review runs page showing LG1/LG3 metadata."""
    rows = ""
    if review_data:
        for key, value in review_data.items():
            value_text = _esc(value)
            if value is True:
                value_text = '<span style="color: green;">✓ true</span>'
            elif value is False:
                value_text = '<span style="color: red;">✗ false</span>'
            rows += f"<tr><td>{_esc(key)}</td><td>{value_text}</td></tr>"
    else:
        rows = "<tr><td colspan='2'>No review data available</td></tr>"

    content = f"""
    <h2>Review Run Status</h2>
    <p><strong>Directory:</strong> <code>{_esc(artifact_dir)}</code></p>
    <table>
      <thead><tr><th>Field</th><th>Value</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    """

    return html_page("Review Runs", content, nav_items, "Review Runs")


def disease_maps_template(diseases: list, nav_items: list) -> str:
    """Disease maps browser page."""
    if not diseases:
        content = "<p>No disease maps found.</p>"
    else:
        rows = ""
        for disease in diseases:
            name = disease.get("disease_name", "unknown")
            therapeutic_area = disease.get("therapeutic_area") or "—"
            program_count = disease.get("program_count", 0)
            cluster_count = disease.get("cluster_count", 0)
            feature_count = disease.get("feature_count", 0)
            rows += f"""
            <tr>
              <td>{_esc(name)}</td>
              <td>{_esc(therapeutic_area)}</td>
              <td>{_esc(program_count)}</td>
              <td>{_esc(cluster_count)}</td>
              <td>{_esc(feature_count)}</td>
            </tr>
            """

        content = f"""
        <h2>Disease Map Index</h2>
        <p>Browse disease maps from the Scientific Cartography review artifacts.</p>
        <table>
          <thead>
            <tr>
              <th>Disease Name</th>
              <th>Therapeutic Area</th>
              <th>Programs</th>
              <th>Clusters</th>
              <th>Features</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
        <p style="font-size: 0.9em; color: #666; margin-top: 10px;">
          Note: Listed in source order. No ranking or prioritization applied.
        </p>
        """

    return html_page("Disease Maps", content, nav_items, "Disease Maps")


def human_decisions_template(decisions: list, nav_items: list) -> str:
    """Human decisions audit trail page."""
    governance_violation = any(d.get("automation_approval") for d in decisions)
    violation_html = ""
    if governance_violation:
        violation_html = '<div class="error"><strong>GOVERNANCE VIOLATION:</strong> automation_approval should be false in all records.</div>'

    if not decisions:
        content = "<p>No human decisions recorded.</p>"
    else:
        rows = ""
        for decision in decisions:
            timestamp = decision.get("created_at_utc", "unknown")
            state = decision.get("decision_state", "unknown")
            actor = decision.get("decision_actor", "unknown")
            reason = decision.get("decision_reason", "—")
            approved = decision.get("review_continuation_approved", False)
            automation = decision.get("automation_approval", False)

            approved_text = '<span style="color: green;">✓</span>' if approved else '<span style="color: red;">✗</span>'
            automation_text = '<span style="color: red;">✗ false</span>'
            if automation:
                automation_text = '<span style="color: red; font-weight: bold;">✗ TRUE (VIOLATION)</span>'

            rows += f"""
            <tr>
              <td><code>{_esc(timestamp)}</code></td>
              <td>{_esc(state)}</td>
              <td>{_esc(actor)}</td>
              <td>{_esc(reason)}</td>
              <td>{approved_text}</td>
              <td>{automation_text}</td>
            </tr>
            """

        content = f"""
        <h2>Human Decision Audit Trail</h2>
        {violation_html}
        <p>Review workflow decisions captured via LG2.</p>
        <table>
          <thead>
            <tr>
              <th>Timestamp</th>
              <th>Decision State</th>
              <th>Actor</th>
              <th>Reason</th>
              <th>Review Approved</th>
              <th>Automation Approval</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
        """

    return html_page("Human Decisions", content, nav_items, "Human Decisions")


def scheduled_review_health_template(executions: list, nav_items: list) -> str:
    """Scheduled review (LG3 wrapper) health page."""
    if not executions:
        content = "<p>No scheduled review executions found.</p>"
    else:
        success_count = sum(1 for e in executions if e.get("outcome") == "success")
        failure_count = sum(1 for e in executions if e.get("outcome") == "failure")

        rows = ""
        for execution in executions:
            timestamp = execution.get("executed_at_utc", "unknown")
            outcome = execution.get("outcome", "unknown")
            duration = execution.get("duration_seconds", 0)
            error_msg = execution.get("error_message", "")
            non_blocking = execution.get("governance", {}).get("non_blocking", True)
            blocking_text = "No (non-blocking)" if non_blocking else "Yes"

            outcome_icon = "✓" if outcome == "success" else "⚠"
            outcome_color = "green" if outcome == "success" else "orange"
            error_text = (
                f"<details><summary>View error</summary><pre>{_esc(error_msg[:200])}{'\u2026' if len(error_msg) > 200 else ''}</pre></details>"
                if error_msg
                else "—"
            )

            rows += f"""
            <tr>
              <td><code>{_esc(timestamp)}</code></td>
              <td><span style="color: {outcome_color};">{outcome_icon} {_esc(outcome)}</span></td>
              <td>{blocking_text}</td>
              <td>{duration:.2f}s</td>
              <td>{error_text}</td>
            </tr>
            """

        content = f"""
        <h2>Scheduled Review Health</h2>
        <p>LG3 wrapper execution audit trail (non-blocking failures are diagnostic only).</p>
        <div class="info">
          <strong>Summary:</strong> {success_count} successes, {failure_count} failures (non-blocking).
          All failures are diagnostic and do not block production.
        </div>
        <table>
          <thead>
            <tr>
              <th>Timestamp</th>
              <th>Outcome</th>
              <th>Blocking</th>
              <th>Duration</th>
              <th>Error (if any)</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
        """

    return html_page("Scheduled Review Health", content, nav_items, "Scheduled Review")


def governance_template(nav_items: list) -> str:
    """Governance boundaries page."""
    content = """
    <h2>Governance Boundaries</h2>
    <p>This dashboard operates under strict governance constraints to ensure read-only, diagnostic operation.</p>
    <p><strong>No scoring is performed.</strong> This is an artifact browser, not an investment decision system.</p>

    <h3>Governance Flags</h3>
    <div class="governance">
      <div class="governance-flag">
        <div class="governance-flag-name">READ_ONLY_DIAGNOSTIC:</div>
        <div class="governance-flag-value flag-true">✓ true</div>
      </div>
      <div class="governance-flag">
        <div class="governance-flag-name">PRODUCTION_MODEL_CHANGE:</div>
        <div class="governance-flag-value flag-false">✗ false</div>
      </div>
      <div class="governance-flag">
        <div class="governance-flag-name">RANKER_CHANGE:</div>
        <div class="governance-flag-value flag-false">✗ false</div>
      </div>
      <div class="governance-flag">
        <div class="governance-flag-name">SELECTOR_CHANGE:</div>
        <div class="governance-flag-value flag-false">✗ false</div>
      </div>
      <div class="governance-flag">
        <div class="governance-flag-name">SIZING_CHANGE:</div>
        <div class="governance-flag-value flag-false">✗ false</div>
      </div>
      <div class="governance-flag">
        <div class="governance-flag-name">FINAL_SCORE_CHANGE:</div>
        <div class="governance-flag-value flag-false">✗ false</div>
      </div>
      <div class="governance-flag">
        <div class="governance-flag-name">TRADING_OR_PORTFOLIO_ACTION:</div>
        <div class="governance-flag-value flag-false">✗ false</div>
      </div>
      <div class="governance-flag">
        <div class="governance-flag-name">AUTOMATION_APPROVAL:</div>
        <div class="governance-flag-value flag-false">✗ false</div>
      </div>
    </div>

    <h3>Key Principles</h3>
    <ul>
      <li><strong>Read-only:</strong> No writes to production systems, no data mutations.</li>
      <li><strong>Artifact-only:</strong> Data sourced from committed artifacts only.</li>
      <li><strong>No scoring:</strong> No computation of alpha, scores, or rankings.</li>
      <li><strong>No automation:</strong> No automated deployment, approval, or trading decisions.</li>
      <li><strong>Diagnostic:</strong> For human review and research only.</li>
    </ul>

    <h3>Disclaimer</h3>
    <div class="info">
      <p>
        This dashboard is an artifact browser for Scientific Cartography review outputs.
        It does not perform scoring, ranking, or portfolio action recommendation.
        No investment recommendation is made.
        No production pipeline action is authorized.
      </p>
    </div>
    """

    return html_page("Governance", content, nav_items, "Governance")
