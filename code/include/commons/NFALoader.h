/*
 * NFALoader.h
 *
 *  Created on: Apr 29, 2018
 *      Author: hyliu
 */

#ifndef NFALOADER_H_
#define NFALOADER_H_

#include <memory>
#include "NFA.h"
#include <string>
#include "pugixml/pugixml.hpp"
//#include "mnrl.hpp"

using std::string;

NFA *load_nfa_from_anml(string filename, bool remove_self_loop = false);

//NFA *load_nfa_from_mnrl(string filename);

NFA *load_nfa_from_file(string filename, bool remove_self_loop = false);


#endif /* NFALOADER_H_ */


