#ifndef KENLM_RS_WRAPPER_H
#define KENLM_RS_WRAPPER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct KenlmModel KenlmModel;

typedef struct KenlmConfig {
  int32_t load_method;
  int32_t arpa_complain;
  float probing_multiplier;
  float unknown_missing_logprob;
  uint8_t show_progress;
} KenlmConfig;

typedef struct KenlmFullScore {
  float prob;
  uint8_t ngram_length;
  uint8_t independent_left;
  uint64_t extend_left;
  float rest;
} KenlmFullScore;

void kenlm_config_default(KenlmConfig *config);

KenlmModel *kenlm_model_load(const char *path, const KenlmConfig *config);
void kenlm_model_free(KenlmModel *model);

const char *kenlm_last_error(void);

size_t kenlm_model_state_size(const KenlmModel *model);
uint8_t kenlm_model_order(const KenlmModel *model);

void kenlm_model_begin_sentence_write(const KenlmModel *model, void *state);
void kenlm_model_null_context_write(const KenlmModel *model, void *state);

uint32_t kenlm_model_index(const KenlmModel *model, const char *word);
int kenlm_model_try_index(const KenlmModel *model, const char *word, uint32_t *out);
uint32_t kenlm_model_begin_sentence_index(const KenlmModel *model);
uint32_t kenlm_model_end_sentence_index(const KenlmModel *model);
uint32_t kenlm_model_not_found_index(const KenlmModel *model);

float kenlm_model_base_score(
    const KenlmModel *model,
    const void *in_state,
    uint32_t word,
    void *out_state);
int kenlm_model_try_base_score(
    const KenlmModel *model,
    const void *in_state,
    uint32_t word,
    void *out_state,
    float *out);

KenlmFullScore kenlm_model_base_full_score(
    const KenlmModel *model,
    const void *in_state,
    uint32_t word,
    void *out_state);
int kenlm_model_try_base_full_score(
    const KenlmModel *model,
    const void *in_state,
    uint32_t word,
    void *out_state,
    KenlmFullScore *out);

int kenlmrs_build_binary_main(int argc, char **argv);
int kenlmrs_cat_compressed_main(int argc, char **argv);
int kenlmrs_count_ngrams_main(int argc, char **argv);
int kenlmrs_dump_counts_main(int argc, char **argv);
int kenlmrs_filter_main(int argc, char **argv);
int kenlmrs_fragment_main(int argc, char **argv);
int kenlmrs_interpolate_main(int argc, char **argv);
int kenlmrs_lmplz_main(int argc, char **argv);
int kenlmrs_phrase_table_vocab_main(int argc, char **argv);
int kenlmrs_query_main(int argc, char **argv);
int kenlmrs_streaming_example_main(int argc, char **argv);

#ifdef __cplusplus
}
#endif

#endif
