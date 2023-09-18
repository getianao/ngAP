#ifndef PPOPP12_CONFIG
#define PPOPP12_CONFIG

#include "commons/common_func.h"

class ppopp12_config : public common_gpunfa_options {
public:
    ppopp12_config() : common_gpunfa_options() {

         auto additional_parser =
            Opt(validation,
                "validation")["--validation"]("validation");
        parser = parser | additional_parser;
    }

    bool validation;


};


#endif
