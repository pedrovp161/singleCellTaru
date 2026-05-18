library(Seurat)
library(dplyr)


QualityControl <- setRefClass(
  "QualityControl",

  methods = list(

    is_outlier = function(x, k = 4, method = "both") {

      x <- as.numeric(x)

      if (all(is.na(x))) {
        return(rep(FALSE, length(x)))
      }

      med <- median(x, na.rm = TRUE)
      dev <- x - med
      mad <- median(abs(dev), na.rm = TRUE)

      if (is.na(mad) || mad == 0) {
        return(rep(FALSE, length(x)))
      }

      threshold <- k * mad

      if (method == "high") {
        return(dev > threshold)
      } else if (method == "low") {
        return(dev < -threshold)
      } else {
        return(abs(dev) > threshold)
      }
    },

    anndataFilters = function(obj,
                              counts_outliers = FALSE,
                              genes_outliers = FALSE,
                              genes_and_counts_outliers = FALSE,
                              mt_percentage_outliers = FALSE,
                              k = 4) {

      stats <- list(
        initial_n_cells = ncol(obj),
        initial_n_genes = nrow(obj)
      )

      filters <- list()

      # -----------------------------
      # COMBINED FILTER
      # -----------------------------
      if (genes_and_counts_outliers) {

        filters[["combined"]] <- function(obj) {
          meta <- obj[[]]

          !(
            .self$is_outlier(meta$nCount_RNA, k = k, method = "low") |
            .self$is_outlier(meta$nFeature_RNA, k = k, method = "low")
          )
        }

      } else {

        if (counts_outliers) {
          filters[["counts"]] <- function(obj) {
            meta <- obj[[]]
            !.self$is_outlier(meta$nCount_RNA, k = k, method = "low")
          }
        }

        if (genes_outliers) {
          filters[["genes"]] <- function(obj) {
            meta <- obj[[]]
            !.self$is_outlier(meta$nFeature_RNA, k = k, method = "low")
          }
        }
      }

      # -----------------------------
      # MT FILTER
      # -----------------------------
      if (mt_percentage_outliers) {

        filters[["mt"]] <- function(obj) {
          meta <- obj[[]]

          if (!"percent.mt" %in% colnames(meta)) {
            warning("percent.mt not found. Skipping MT filter.")
            return(rep(TRUE, ncol(obj)))
          }

          !.self$is_outlier(meta$percent.mt, k = k, method = "high")
        }
      }

      # -----------------------------
      # APPLY FILTERS SAFELY
      # -----------------------------
      for (nm in names(filters)) {

        mask <- filters[[nm]](obj)
        keep <- colnames(obj)[mask]

        message(nm, " -> keeping ", length(keep), " / ", ncol(obj))

        if (length(keep) == 0) {
          warning(paste("Filter", nm, "would remove all cells. Skipping."))
          next
        }

        obj <- subset(obj, cells = keep)

        stats[[paste0("n_after_", nm)]] <- ncol(obj)
      }

      stats$final_n_cells <- ncol(obj)
      stats$final_n_genes <- nrow(obj)

      obj@misc$preprocessing_stats <- stats

      return(obj)
    },

    preprocessing = function(obj,
                             genes_and_counts_outliers = TRUE,
                             mt_percentage_outliers = TRUE,
                             k = 4) {

      obj <- UpdateSeuratObject(obj)

      obj[["percent.mt"]] <- PercentageFeatureSet(obj, pattern = "^MT-")

      obj <- .self$anndataFilters(
        obj,
        genes_and_counts_outliers = genes_and_counts_outliers,
        mt_percentage_outliers = mt_percentage_outliers,
        k = k
      )

      return(obj)
    }
  )
)


