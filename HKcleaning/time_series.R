# LTG_bypatient.csv imported
# run in Rstudio
df <- NTG_bypatient
venn3 <- list(
    CT = df$patient_id[df$ct == 1],
    Oph = df$patient_id[df$oph_total > 0],
    OCT = df$patient_id[df$OCT > 0]
  )

# overlap count
library(dplyr)
library(lubridate)

ct_time <- plot_df |> 
  filter(type == "CT") |>
  group_by(patient_id) |>
  summarize(
    n_ct = n(),
    ct_start = min(date),
    ct_end = max(date)
  )

oct_time <- plot_df |>
  filter(type == "OCT") |>
  group_by(patient_id) |>
  summarize(
    n_oct = n(),
    oct_start = min(date),
    oct_end = max(date)
  )

overlap_df <- inner_join(ct_time, oct_time, by = "patient_id") |>
  filter(n_ct > 1, n_oct > 1) |>
  mutate(
    overlap = !(ct_end < oct_start | oct_end < ct_start)
  )

summary_table <- overlap_df |>
  count(overlap)

print(summary_table)

library(tidyverse)
library(lubridate)

# -------------------------------
# Input: df = LTG/NTG bypatient dataframe
# -------------------------------
df_sub <- df |>
  as_tibble() |>
  filter(ct == 1 & oph_total > 0)

# -------------------------------
# Function to expand a date column into long format
# -------------------------------
expand_dates <- function(data, id_col, date_col, type_label) {
  data |>
    select(all_of(c(id_col, date_col))) |>
    filter(!is.na(.data[[date_col]])) |>
    separate_rows(all_of(date_col), sep = "\\|") |>
    mutate(
      date = ymd(.data[[date_col]]),
      type = type_label
    ) |>
    select(patient_id = all_of(id_col), date, type)
}

# -------------------------------
# Expand CT + all ophthalmic tag _dates columns
# -------------------------------
ct_long <- expand_dates(df_sub, "patient_id", "ct_dates", "CT")

oph_date_cols <- grep("_dates$", names(df_sub), value = TRUE)
oph_long <- map_dfr(oph_date_cols, function(col) {
  tag <- sub("_dates$", "", col)
  expand_dates(df_sub, "patient_id", col, tag)
})

# Combine into one long-format timeline df
plot_df <- bind_rows(ct_long, oph_long)

# -------------------------------
# Create consistent patient ordering by CT start date
# -------------------------------
patient_order <- plot_df |>
  filter(type == "CT") |>
  group_by(patient_id) |>
  summarize(first_ct = min(date), .groups = "drop") |>
  arrange(first_ct) |>
  pull(patient_id)

plot_df$patient_id <- factor(plot_df$patient_id, levels = patient_order)

# -------------------------------
# Assign batch group (e.g., 82 patients per plot)
# -------------------------------
patients_grouped <- tibble(patient_id = levels(plot_df$patient_id)) |>
  mutate(group = ceiling(row_number() / 82))

plot_df <- left_join(plot_df, patients_grouped, by = "patient_id")

# -------------------------------
# Plotting function for a single batch
# -------------------------------
plot_batch <- function(batch_id) {
  df_batch <- filter(plot_df, group == batch_id)
  
  ggplot(df_batch, aes(x = date, y = patient_id, color = type)) +
    geom_line(
      data = filter(df_batch, type == "CT"),
      aes(group = patient_id),
      color = "red", linewidth = 0.6, alpha = 0.3
    ) +
    geom_line(
      data = filter(df_batch, type == "OCT"),
      aes(group = patient_id),
      color = "blue", linewidth = 0.6, alpha = 0.3
    ) +
    geom_point(
      aes(shape = ifelse(type %in% c("CT", "OCT"), "circle", "x")),
      size = 1.2, alpha = 0.7
    ) +
    scale_shape_manual(values = c("circle" = 21, "x" = 4), guide = "none") +
    scale_color_manual(values = c("CT" = "red", "OCT" = "blue")) +
    scale_x_date(date_breaks = "6 months", date_labels = "%Y-%m") +
    labs(
      title = paste("Timeline for Patients", (batch_id - 1) * 82 + 1, "to", batch_id * 82),
      x = "Date", y = "Patient ID", color = "Modality"
    ) +
    theme_minimal(base_size = 10) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 6),
      axis.text.y = element_text(size = 3),
      plot.margin = margin(10, 10, 10, 10)
    ) +
    coord_cartesian(clip = "off")
}

# -------------------------------
# To display or save plots
# -------------------------------
# Show batch 1 in viewer:
# plot_batch(1)

# Or save all batches:
pdf("ntg_timeline_batches.pdf", width = 12, height = 18)

for (g in sort(unique(as.numeric(plot_df$group)))) {
  print(plot_batch(g))
}

dev.off()