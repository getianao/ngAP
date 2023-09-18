#ifndef INFANT_CONFIG
#define INFANT_CONFIG

#include "commons/common_func.h"

using namespace clara;

class infant_config : public common_gpunfa_options {
public:
    infant_config() : common_gpunfa_options()               
    {
        this->num_state_per_group = this->block_size;

        auto additional_parser =
            Opt(num_state_per_group,
                "num_state_per_group")["--num-state-per-group"](
                "number of state per group in infant. ") |
            Opt(validation,
                "validation")["--validation"]("fake validation (do nothing)");

        parser = parser | additional_parser;
    }

    int num_state_per_group;
    bool validation;
};


#endif





