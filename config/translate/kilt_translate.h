#ifdef KILT_CONFIG_TRANSLATE_CK
#include "config/translate/ck/translation_table.h"
#elif KILT_CONFIG_TRANSLATE_X
#include "config/translate/x/translation_table.h"
#else
#error Config translation table not defined.
#endif