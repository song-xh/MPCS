# Issue tracker: Local Markdown

Issues and specs for this repo live as Markdown files in `.scratch/`.

## Conventions

- One feature per directory: `.scratch/<feature-slug>/`
- The spec is `.scratch/<feature-slug>/spec.md`
- Implementation issues are one file per ticket at `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, numbered from `01`
- Triage state is recorded as a `Status:` line near the top of each issue file, using the strings in `triage-labels.md`
- Comments append under a `## Comments` heading

## Skill operations

- To publish an issue or spec, create the appropriate file under `.scratch/<feature-slug>/`.
- To fetch a ticket, read its referenced path or issue number.

## Wayfinding operations

- Map: `.scratch/<effort>/map.md`, with Notes, Decisions-so-far, and Fog.
- Child ticket: `.scratch/<effort>/issues/NN-<slug>.md`. Record its `Type:` (`research`, `prototype`, `grilling`, or `task`) and `Status:` (`claimed` or `resolved`).
- Blocking: list dependencies as `Blocked by: NN, NN`. A ticket is unblocked when every listed ticket is resolved.
- Frontier: choose the first numbered open, unblocked, unclaimed ticket.
- Claim: set `Status: claimed` before work.
- Resolve: append an answer under `## Answer`, set `Status: resolved`, and add a context pointer to the map’s Decisions-so-far.
