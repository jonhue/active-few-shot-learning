#!/usr/bin/env python3
from pathlib import Path
import shutil
import textwrap

from jinja2 import Environment
from jinja2 import FileSystemLoader
from markupsafe import Markup
import pygments.formatters.html
import pygments.lexers.python

import pdoc.render

here = Path(__file__).parent

if __name__ == "__main__":
    demo1 = here / "demo1.py"
    demo2 = here / "demo2.py"
    env = Environment(
        loader=FileSystemLoader([here, here / ".." / "pdoc" / "templates"]),
        autoescape=True,
    )

    lexer = pygments.lexers.python.PythonLexer()
    formatter = pygments.formatters.html.HtmlFormatter(style="dracula")
    pygments_css = formatter.get_style_defs()
    example_html1 = Markup(pygments.highlight(demo1.read_text("utf8"), lexer, formatter).replace('converged', '<span class="highlighted">converged</span>'))
    example_html2 = Markup(pygments.highlight(demo2.read_text("utf8"), lexer, formatter))

    (here / "index.html").write_bytes(
        env.get_template("index.html.jinja2")
        .render(
            example_html1=example_html1,
            example_html2=example_html2,
            pygments_css=pygments_css,
            __version__=pdoc.__version__,
        )
        .encode()
    )

    if (here / "docs").is_dir():
        shutil.rmtree(here / "docs")

    # Render main docs
    pdoc.render.configure(
        edit_url_map={
            "pdoc": "https://github.com/mitmproxy/pdoc/blob/main/pdoc/",
            "demo": "https://github.com/mitmproxy/pdoc/blob/main/test/testdata/demo",
            "demo_long": "https://github.com/mitmproxy/pdoc/blob/main/test/testdata/demo_long",
        },
        favicon="/favicon.svg",
        logo="/logo.svg",
        logo_link="https://pdoc.dev",
        footer_text=f"pdoc {pdoc.__version__}",
    )

    # Add sitemap.xml
    with (here / "sitemap.xml").open("w", newline="\n") as f:
        f.write(
            textwrap.dedent(
                """
        <?xml version="1.0" encoding="utf-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
           xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
           xsi:schemaLocation="http://www.sitemaps.org/schemas/sitemap/0.9 http://www.sitemaps.org/schemas/sitemap/0.9/sitemap.xsd">
        """
            ).strip()
        )
        for file in here.glob("**/*.html"):
            if file.name.startswith("_"):
                continue
            filename = str(file.relative_to(here).as_posix()).replace("index.html", "")
            f.write(f"""\n<url><loc>https://pdoc.dev/{filename}</loc></url>""")
        f.write("""\n</urlset>""")
