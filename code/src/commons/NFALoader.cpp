/*
 * NFALoader.cpp
 *
 *  Created on: Apr 29, 2018
 *      Author: hyliu
 */

#include "NFALoader.h"
#include "NFA.h"
#include <memory>
#include <string>
#include <iostream>
#include "pugixml.hpp"

using std::cout;
using std::endl;
using std::unique_ptr;
using std::string;
using std::shared_ptr;
using std::cerr;
using std::dynamic_pointer_cast;

static int to_start_enum(string start) {
    int st = 0;
	if(start.compare("start-of-data") == 0) {
        st = NODE_START_ENUM::START;
    } else if (start.compare("all-input") == 0) {
    	st = NODE_START_ENUM::START_ALWAYS_ENABLED;
    } else {
    	st = 0;
    }

    return st;
}


static Node *parse_new_state(pugi::xml_node ste) {
    Node *ptr = new Node();
    string id;
    string symbol_set;
    string start;

    // gather attributes
    for (pugi::xml_attribute attr = ste.first_attribute(); attr; attr = attr.next_attribute()) {

        string str = attr.name();

        if(str.compare("id") == 0) {
            id = attr.value();
        } else if(str.compare("symbol-set") == 0) {
            symbol_set = attr.value();
        }else if(str.compare("start") == 0) {
            start = attr.value();
            //cout <<"start" << endl;
        }
    }


    ptr->symbol_set_str = symbol_set;
    ptr->start = to_start_enum(start);
    ptr->str_id = id;
    ptr->original_id = id;

    ptr->symbol_set_to_bit();

    return ptr;
}


NFA *load_nfa_from_anml(string filename) {

    pugi::xml_document doc;
    if (!doc.load_file(filename.c_str(),pugi::parse_default|pugi::parse_declaration)) {
        cout << "Could not load .xml file: " << filename << endl;
        exit(1);
    } else {
    	cout << "load automata from file = " << filename << endl;
    }

    NFA *nfa = new NFA();
    
    // can handle finding automata-network at one or two layers under root


    pugi::xml_node nodes = doc.child("anml").child("automata-network");
    

    //cout << "nodename = " << nodes.name() << endl;

    string nodename = nodes.name();
    nodename.erase(nodename.find_last_not_of(" \n\r\t")+1);

    if (nodename == "") {
        nodes = doc.child("automata-network");
        nodename =  nodes.name();
        nodename.erase(nodename.find_last_not_of(" \n\r\t")+1);
        //cout << "nodename2 = " << nodes.name() << endl;
    }
    


    string id_tmp = nodes.attribute("id").value();

    for (pugi::xml_node node = nodes.first_child(); node; node = node.next_sibling()) {

        string str = node.name();
        //cout << "nodename = "<< str << endl;

        if(str.compare("state-transition-element") == 0) {
            nfa->addNode(parse_new_state(node));
            Node *current_node = nfa->get_node_by_int_id(nfa->size() - 1);

            for(pugi::xml_node aom : node.children("activate-on-match")) {
            	nfa->addEdge(current_node->str_id, aom.attribute("element").value());
            }

            for(pugi::xml_node aom : node.children("report-on-match")) {
                //s->setReporting(true);
            	current_node->report = true;
            }

            for(pugi::xml_node aom : node.children("report-on-match")) {
                current_node->report_code = aom.attribute("reportcode").value();
            }


        } else if(str.compare("and") == 0) {
        	 cout << "NODE: " << str << " NOT IMPLEMENTED IN PARSER..." << endl;
             exit(-1);
        } else if(str.compare("or") == 0) {
        	 cout << "NODE: " << str << " NOT IMPLEMENTED IN PARSER..." << endl;
             exit(-1);
        }else if(str.compare("nor") == 0) {
        	 cout << "NODE: " << str << " NOT IMPLEMENTED IN PARSER..." << endl;
             exit(-1);
        } else if(str.compare("counter") == 0) {
        	 cout << "NODE: " << str << " NOT IMPLEMENTED IN PARSER..." << endl;
             exit(-1);
        } else if (str.compare("inverter") == 0) {
        	 cout << "NODE: " << str << " NOT IMPLEMENTED IN PARSER..." << endl;
             exit(-1);
        } else if(str.compare("description") == 0) {
            // DO NOTHING
        } else {
            cout << "NODE: " << str << " NOT IMPLEMENTED IN PARSER..." << endl;
            exit(-1);
        }

    }



	return nfa;
}



static bool has_suffix(const std::string &str, const std::string &suffix)
{
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

NFA *load_nfa_from_file(string filename) {
    if (has_suffix(filename, ".anml")) {
        return load_nfa_from_anml(filename);
    } else {
        cout << "Unsupported NFA file type: " << filename << endl;
        exit(-1);
    }
}


