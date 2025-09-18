# ---- setup ----
library(tidyverse)
library(rsample)
library(randomForest)
library(caret)
library(pROC)
library(kableExtra)
library(randomForestExplainer)
library(corrr)
set.seed(42)

# Folders for Overleaf-ready outputs
dir.create("export/figs", recursive = TRUE, showWarnings = FALSE)
dir.create("export/tables", recursive = TRUE, showWarnings = FALSE)
dir.create("export/models", recursive = TRUE, showWarnings = FALSE)
# ---- data ----
data <- readr::read_csv("~/Desktop/UNGDC/data/processed/liwc_meta.csv") |>
  dplyr::select(-"...1")

# outcome as factor; keep just the columns you actually need
# (adjust these identifiers to match your file)
data <- data |>
  dplyr::mutate(
    year = as.numeric(year),
    dd_democracy = factor(dd_democracy)
  )

# a single preprocessor used for both train & test
prep_df <- function(df) {
  df |>
    dplyr::select(-ccode_iso, -session) |>
    # rename reserved words & drop raw text
    dplyr::rename(function_liwc = `function`) |>
    dplyr::select(-text)
}

# stratified split (once)
split <- initial_split(data, prop = 0.7, strata = "dd_democracy")
train <- training(split) |> prep_df() |> drop_na(dd_democracy)
test  <- testing(split)  |> prep_df()


# ---- keep only LIWC block WC..OtherP ----
train_liwc <- train %>%
  dplyr::select(dd_democracy, WC:OtherP) %>%
  tidyr::drop_na(dd_democracy)

test_liwc <- test %>%
  dplyr::select(dd_democracy, WC:OtherP)
````



# ---- models ----
# Baseline logistic (glm binomial)
logit1 <- glm(dd_democracy ~ ., data = train_liwc, family = binomial())

# and then random forest
p <- ncol(train_liwc) - 1
rf1 <- randomForest(dd_democracy ~ ., data = train_liwc,
                    ntree = 500, mtry = p, importance = TRUE)


saveRDS(logit1, "export/models/logit1.rds")
saveRDS(rf1,    "export/models/rf1.rds")

# ---- evaluation ----
export_table <- function(df, fname_csv, fname_tex, caption) {
  readr::write_csv(df, file.path("export/tables", fname_csv))
  tab <- kbl(df, format = "latex", booktabs = TRUE, caption = caption) |>
    kable_styling(latex_options = c("hold_position", "scale_down"))
  save_kable(tab, file.path("export/tables", fname_tex))
}

# Predictions
pred_logit_prob <- predict(logit1, newdata = test_liwc, type = "response")
pred_logit_cls  <- factor(ifelse(pred_logit_prob >= 0.5, "1", "0"),
                          levels = levels(test_liwc$dd_democracy))

pred_rf_prob <- predict(rf1, newdata = test_liwc, type = "prob")[, "1"]
pred_rf_cls  <- predict(rf1, newdata = test_liwc, type = "class")

# Confusion matrices
cm_rf    <- caret::confusionMatrix(test_liwc$dd_democracy, pred_rf_cls,    positive = "1")
cm_logit <- caret::confusionMatrix(test_liwc$dd_democracy, pred_logit_cls, positive = "1")

export_table(as.data.frame(cm_rf$table),
             "confusion_rf.csv", "confusion_rf.tex",
             "Confusion matrix — Random Forest")

export_table(as.data.frame(cm_logit$table),
             "confusion_logit.csv", "confusion_logit.tex",
             "Confusion matrix — Logistic regression")

# ROC/AUC (both models)
roc_rf    <- pROC::roc(response = test_liwc$dd_democracy, predictor = pred_rf_prob)
roc_logit <- pROC::roc(response = test_liwc$dd_democracy, predictor = pred_logit_prob)

readr::write_csv(
  tibble(model = c("RF","Logit"),
         AUC   = c(as.numeric(pROC::auc(roc_rf)), as.numeric(pROC::auc(roc_logit)))),
  "export/tables/auc_summary.csv"
)

# Publication figure: both curves
png("export/figs/roc_rf_logit.png", width = 2000, height = 1400, res = 300)
plot(roc_logit, col = "gray30", lwd = 2, legacy.axes = FALSE,
     main = "ROC — Logistic vs. Random Forest")
plot(roc_rf, col = "blue", lwd = 2, add = TRUE)
legend("bottomright",
       legend = c(paste0("Logit AUC=", round(pROC::auc(roc_logit), 3)),
                  paste0("RF AUC=",    round(pROC::auc(roc_rf),    3))),
       lwd = 2, col = c("gray30", "blue"), bty = "n")
dev.off()

pdf("export/figs/roc_rf_logit.pdf", width = 6.5, height = 4.5)
plot(roc_logit, col = "gray30", lwd = 2, legacy.axes = FALSE,
     main = "ROC — Logistic vs. Random Forest")
plot(roc_rf, col = "blue", lwd = 2, add = TRUE)
legend("bottomright",
       legend = c(paste0("Logit AUC=", round(pROC::auc(roc_logit), 3)),
                  paste0("RF AUC=",    round(pROC::auc(roc_rf),    3))),
       lwd = 2, col = c("gray30", "blue"), bty = "n")
dev.off()



