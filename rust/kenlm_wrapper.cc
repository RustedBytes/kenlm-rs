#include "kenlm_wrapper.h"

#include "lm/config.hh"
#include "lm/model.hh"
#include "lm/return.hh"
#include "lm/virtual_interface.hh"
#include "util/exception.hh"
#include "util/mmap.hh"

#include <exception>
#include <memory>
#include <new>
#include <string>

struct KenlmModel {
  lm::base::Model *model;
};

namespace {
thread_local std::string g_last_error;

void ClearLastError() { g_last_error.clear(); }

void StoreException(const char *context) {
  try {
    throw;
  } catch (const std::exception &exception) {
    g_last_error = std::string(context) + ": " + exception.what();
  } catch (...) {
    g_last_error = std::string(context) + ": unknown C++ exception";
  }
}

lm::ngram::Config ToKenlmConfig(const KenlmConfig *input) {
  lm::ngram::Config config;
  if (!input) return config;

  config.load_method = static_cast<util::LoadMethod>(input->load_method);
  config.arpa_complain =
      static_cast<lm::ngram::Config::ARPALoadComplain>(input->arpa_complain);
  config.probing_multiplier = input->probing_multiplier;
  config.unknown_missing_logprob = input->unknown_missing_logprob;
  config.show_progress = input->show_progress != 0;
  if (!input->show_progress) {
    config.messages = NULL;
  }
  return config;
}

KenlmFullScore ToCFullScore(const lm::FullScoreReturn &input) {
  KenlmFullScore output;
  output.prob = input.prob;
  output.ngram_length = input.ngram_length;
  output.independent_left = input.independent_left ? 1 : 0;
  output.extend_left = input.extend_left;
  output.rest = input.rest;
  return output;
}
}  // namespace

extern "C" {

void kenlm_config_default(KenlmConfig *config) {
  if (!config) return;
  lm::ngram::Config defaults;
  config->load_method = static_cast<int32_t>(defaults.load_method);
  config->arpa_complain = static_cast<int32_t>(defaults.arpa_complain);
  config->probing_multiplier = defaults.probing_multiplier;
  config->unknown_missing_logprob = defaults.unknown_missing_logprob;
  config->show_progress = defaults.show_progress ? 1 : 0;
}

KenlmModel *kenlm_model_load(const char *path, const KenlmConfig *config) {
  ClearLastError();
  try {
    lm::ngram::Config converted = ToKenlmConfig(config);
    std::unique_ptr<lm::base::Model> model(lm::ngram::LoadVirtual(path, converted));
    KenlmModel *handle = new KenlmModel;
    handle->model = model.release();
    return handle;
  } catch (...) {
    StoreException("failed to load KenLM model");
    return NULL;
  }
}

void kenlm_model_free(KenlmModel *model) {
  if (!model) return;
  delete model->model;
  delete model;
}

const char *kenlm_last_error(void) { return g_last_error.c_str(); }

size_t kenlm_model_state_size(const KenlmModel *model) {
  return model->model->StateSize();
}

uint8_t kenlm_model_order(const KenlmModel *model) {
  return model->model->Order();
}

void kenlm_model_begin_sentence_write(const KenlmModel *model, void *state) {
  model->model->BeginSentenceWrite(state);
}

void kenlm_model_null_context_write(const KenlmModel *model, void *state) {
  model->model->NullContextWrite(state);
}

uint32_t kenlm_model_index(const KenlmModel *model, const char *word) {
  return model->model->BaseVocabulary().Index(word);
}

int kenlm_model_try_index(const KenlmModel *model, const char *word, uint32_t *out) {
  ClearLastError();
  try {
    *out = model->model->BaseVocabulary().Index(word);
    return 0;
  } catch (...) {
    StoreException("failed to look up KenLM vocabulary entry");
    return 1;
  }
}

uint32_t kenlm_model_begin_sentence_index(const KenlmModel *model) {
  return model->model->BaseVocabulary().BeginSentence();
}

uint32_t kenlm_model_end_sentence_index(const KenlmModel *model) {
  return model->model->BaseVocabulary().EndSentence();
}

uint32_t kenlm_model_not_found_index(const KenlmModel *model) {
  return model->model->BaseVocabulary().NotFound();
}

float kenlm_model_base_score(
    const KenlmModel *model,
    const void *in_state,
    uint32_t word,
    void *out_state) {
  return model->model->BaseScore(in_state, word, out_state);
}

int kenlm_model_try_base_score(
    const KenlmModel *model,
    const void *in_state,
    uint32_t word,
    void *out_state,
    float *out) {
  ClearLastError();
  try {
    *out = model->model->BaseScore(in_state, word, out_state);
    return 0;
  } catch (...) {
    StoreException("failed to score KenLM state transition");
    return 1;
  }
}

KenlmFullScore kenlm_model_base_full_score(
    const KenlmModel *model,
    const void *in_state,
    uint32_t word,
    void *out_state) {
  return ToCFullScore(model->model->BaseFullScore(in_state, word, out_state));
}

int kenlm_model_try_base_full_score(
    const KenlmModel *model,
    const void *in_state,
    uint32_t word,
    void *out_state,
    KenlmFullScore *out) {
  ClearLastError();
  try {
    *out = ToCFullScore(model->model->BaseFullScore(in_state, word, out_state));
    return 0;
  } catch (...) {
    StoreException("failed to fully score KenLM state transition");
    return 1;
  }
}

}  // extern "C"
