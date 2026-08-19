"""
TALO CLI -- runs the end-to-end pipeline described in Section 5.5 for a
single natural-language analytics query: classification, routing, and
prompt preview, with an optional live model call.

Usage:
    python run_talo.py "Zeige Umsatz je Region"
    python run_talo.py "Show me revenue by region" --live
"""
import argparse

from rich.console import Console

from talo.orchestrator import TALOOrchestrator

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Natural-language analytics query")
    parser.add_argument(
        "--live", action="store_true",
        help="Make a real API call (costs money). Without this flag, TALO "
             "only shows classification, routing, and the rendered prompt.",
    )
    args = parser.parse_args()

    orchestrator = TALOOrchestrator()
    result = orchestrator.run(args.query, dry_run=not args.live)

    console.print(f"\n[bold]Query:[/bold] {result.query}")

    cls = result.classification
    console.print(
        f"\n[bold]Classification[/bold] -> [cyan]{cls.task_class}[/cyan] "
        f"(confidence: {cls.confidence:.2f}"
        f"{', fallback' if cls.used_fallback else ''})"
    )

    if result.routing:
        r = result.routing
        console.print(
            f"[bold]Routing[/bold]        -> model=[cyan]{r.model_id}[/cyan] "
            f"strategy=[yellow]{r.strategy}[/yellow] "
            f"(routing confidence: {r.confidence})"
        )

    if result.error:
        console.print(f"\n[red]Error:[/red] {result.error}")
        return

    if result.prompt:
        console.print("\n[bold]Prompt preview:[/bold]")
        console.print(result.prompt)

    if result.dry_run:
        console.print(
            "\n[yellow]Dry run -- no API call made. Pass --live to call the model.[/yellow]"
        )
    else:
        console.print(f"\n[bold]Output:[/bold]\n{result.output}")


if __name__ == "__main__":
    main()
