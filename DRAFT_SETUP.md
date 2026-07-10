# Draft Assistant — one-time Supabase setup

The Live Draft Tracker stores picks in a Supabase table. Run this once in the
Supabase dashboard (SQL Editor → New query → paste → Run):

```sql
create table if not exists draft_picks (
  id bigint generated always as identity primary key,
  year int not null default 2026,
  round int not null,
  pick int not null,
  team text not null,
  player_name text not null,
  slot_value numeric,
  signing_bonus numeric,
  entered_by text,
  created_at timestamptz not null default now()
);

alter table draft_picks enable row level security;

create policy "draft_picks_read" on draft_picks
  for select to anon using (true);
create policy "draft_picks_insert" on draft_picks
  for insert to anon with check (true);
create policy "draft_picks_update" on draft_picks
  for update to anon using (true) with check (true);
```

That's it — reload the Draft Assistant page and the Live Tracker goes active.

Notes
- Same anon key the Portal Commitment page uses; no code changes needed.
- To wipe a bad pick: delete the row in Supabase Table Editor (no delete from the page by design).
- Reference data (board, trends, history) lives in `data/draft/` — regenerate from the
  workbook by asking Claude to re-run the exporter after you update the Master Document.
