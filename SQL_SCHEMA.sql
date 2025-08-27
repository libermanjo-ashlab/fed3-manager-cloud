
create table if not exists devices (
  id bigserial primary key,
  housing_id text,
  housing_status text,
  electronics_id text,
  electronics_status text,
  status_in_lab text,
  status_with_mice text,
  in_use boolean default false,
  "user" text,
  current_location text,
  exp_start_date text,
  notes text,
  status_bucket text
);
create index if not exists idx_devices_status_bucket on devices(status_bucket);
create index if not exists idx_devices_user on devices("user");
create index if not exists idx_devices_housing on devices(housing_id);
create index if not exists idx_devices_electronics on devices(electronics_id);

create table if not exists inventory (
  id bigserial primary key,
  item text,
  qty double precision default 0
);
create index if not exists idx_inventory_item on inventory(item);

create table if not exists actions (
  id bigserial primary key,
  ts timestamp with time zone default now(),
  actor text,
  action text,
  housing_id text,
  electronics_id text,
  details text
);
create index if not exists idx_actions_ts on actions(ts desc);
