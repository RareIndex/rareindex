# utils/newsletter.py
import html
import pandas as pd
import streamlit as st

def render_newsletter_tools(snippet: str, key: str = "newsletter"):
    """
    Renders: editable text area, Download .txt, Copy-to-clipboard (JS), and light UX polish.
    Pass a unique `key` per usage to avoid widget ID collisions on different tabs/sections.
    Returns the possibly edited snippet text.
    """
    st.markdown("#### ✂️ Newsletter tools")

    editable_snippet = st.text_area(
        "Preview / edit",
        value=snippet or "",
        height=260,
        key=f"{key}-textarea",
    )

    # Disable actions if empty
    disabled = not (editable_snippet and editable_snippet.strip())

    # 1) Download .txt (dated filename) + toast on click
    date_str = pd.Timestamp.today().strftime("%Y-%m-%d")
    clicked = st.download_button(
        label="⬇️ Download .txt",
        data=editable_snippet,
        file_name=f"rare-index-newsletter-{date_str}.txt",
        mime="text/plain",
        help="Save your newsletter snippet as a .txt file",
        disabled=disabled,
        key=f"{key}-download",
    )
    if clicked:
        st.toast("Saved .txt")

    # 2) Copy to clipboard (JS) — shows inline status text; disabled state handled in JS
    # Escape for safe HTML embedding inside the hidden <textarea>
    safe_html = html.escape(editable_snippet or "")

    st.markdown(
        f"""
<textarea id="{key}-src" style="position:absolute; left:-9999px;">{safe_html}</textarea>
<button id="{key}-copy-btn" {'disabled' if disabled else ''}>📋 Copy to clipboard</button>
<span id="{key}-copy-status" style="margin-left:8px; font-size:0.9em; opacity:0.8;"></span>
<script>
  const btn = document.getElementById('{key}-copy-btn');
  const src = document.getElementById('{key}-src');
  const status = document.getElementById('{key}-copy-status');
  const queryLabel = 'textarea[aria-label="Preview / edit"]';
  function findVisibleTextArea() {{
    const areas = window.parent.document.querySelectorAll(queryLabel);
    return areas[areas.length - 1] || null;
  }}
  btn && (btn.onclick = async () => {{
    if (btn.disabled) return;
    try {{
      const ta = findVisibleTextArea();
      if (ta) src.value = ta.value;
      await navigator.clipboard.writeText(src.value);
      status.textContent = "Copied!";
      setTimeout(()=> status.textContent = "", 1500);
    }} catch (e) {{
      status.textContent = "Copy failed";
      setTimeout(()=> status.textContent = "", 1500);
    }}
  }});
</script>
        """,
        unsafe_allow_html=True
    )

    # Small UX hints
    if disabled:
        st.caption("Type or generate content above to enable the buttons.")

    return editable_snippet
