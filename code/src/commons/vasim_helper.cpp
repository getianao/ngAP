#include "vasim_helper.h"
#include <string>
#include <iostream>
#include <bitset>


using std::cout;
using std::endl;

/**
 * from Jack Wadden's VASim
 */

void VASim::find_and_replace(std::string & source, std::string const & find, std::string const & replace) {
    for(std::string::size_type i = 0; (i = source.find(find, i)) != std::string::npos;) {
        source.replace(i, find.length(), replace);
        i += replace.length();
    }
}


/**
 * Sets a range for a given bitset from start to end as value.
 */
void VASim::setRange(std::bitset<256> &column, int start, int end, int value) {

    for(;start <= end; start++){
        column.set(start, value);
    }
}


/**
 * from Jack Wadden's VASim
 */
/**
 * Parses a string representation of a symbol/character set and sets each corresponding bit in column.
 */
// fixed the bug. 
void VASim::parseSymbolSet(std::bitset<256> &column, std::string symbol_set) {
    
    if(symbol_set.compare("*") == 0){
        column.set();
        return;
    }
    
    // KAA found that apcompile parses symbol-set="." to mean "^\x0a"
    // hard-coding this here
    if(symbol_set.compare(".") == 0) {
        column.set('\n',1);
        column.flip();
        return;
    }
        
    bool in_charset = false;
    bool escaped = false;
    bool inverting = false;
    bool range_set = false;
    int bracket_sem = 0;
    int brace_sem = 0;
    const unsigned int value = 1;
    uint32_t last_char = 0;
    uint32_t range_start = 0;

    // SPECIAL CHAR CODES
    uint32_t OPEN_BRACKET = 256;
    
    // handle symbol sets that start and end with curly braces {###}
    if((symbol_set[0] == '{') &&
       (symbol_set[symbol_set.size() - 1] == '}')){
        
        std::cout << "CURLY BRACES NOT IMPLEMENTED" << std::endl;
        exit(1);
    }

    int index = 0;
    while(index < symbol_set.size()) {

        unsigned char c = symbol_set[index];

        //std::cout << "PROCESSING CHAR: " << c << std::endl;
        
        switch(c){
            
            // Brackets
        case '[' :
            if(escaped){
                column.set(c, value);
                if(range_set){
                    setRange(column,range_start,c,value);
                    range_set = false;
                }
                last_char = c;
                escaped = false;
            }else{
                last_char = OPEN_BRACKET;
                bracket_sem++;
            }
            break;
        case ']' :
            if(escaped){
                column.set(c, value);
                if(range_set){
                    setRange(column,range_start,c,value);
                    range_set = false;
                }
                escaped = false;
                last_char = c;
            }else{
                bracket_sem--;
            }
            
            break;

            // Braces
        case '{' :
            
            //if(escaped){
                column.set(c, value);
                if(range_set){
                    setRange(column,range_start,c,value);
                    range_set = false;
                }
                
                last_char = c;
                //escaped = false;
                //}else{
                //brace_sem++;
                //}
            break;
        case '}' :
            //if(escaped){
                column.set(c, value);
                if(range_set){
                    setRange(column,range_start,c,value);
                    range_set = false;
                }
                last_char = c;
                //escaped = false;
                //}else{
                //brace_sem--;
                //}
            break;
            
            //escape
        case '\\' :
            if(escaped){
                column.set(c, value);
                if(range_set){
                    setRange(column,range_start,c,value);
                    range_set = false;
                }

                last_char = c;
                escaped = false;
            }else{
                escaped = true;
            }
            break;
            
            // escaped chars
        case 'n' :
            if(escaped){
                column.set('\n',value);
                if(range_set){
                    setRange(column,range_start,'\n',value);
                    range_set = false;
                }
                last_char = '\n';
                escaped = false;
            }else{
                column.set(c, value);
                if(range_set){
                    setRange(column,range_start,c,value);
                    range_set = false;
                }
                last_char = c;
            }
            break;
        case 'r' :
            if(escaped){
                column.set('\r',value);
                if(range_set){
                    setRange(column,range_start,'\r',value);
                    range_set = false;
                }
                last_char = '\r';
                escaped = false;
            }else{
                column.set(c, value);
                if(range_set){
                    setRange(column,range_start,c,value);
                    range_set = false;
                }
                last_char = c;
            }
            break;
        case 't' :
            if(escaped){
                column.set('\t',value);
                if(range_set){
                    setRange(column,range_start,'\t',value);
                    range_set = false;
                }
                last_char = '\t';
                escaped = false;
            }else{
                column.set(c, value);
                if(range_set){
                    setRange(column,range_start,c,value);
                    range_set = false;
                }
                last_char = c;
            }
            break;
        case 'a' :
            if(escaped){
                column.set('\a',value);
                if(range_set){
                    setRange(column,range_start,'\a',value);
                    range_set = false;
                }
                last_char = '\a';
                escaped = false;
            }else{
                column.set(c, value);
                if(range_set){
                    setRange(column,range_start,c,value);
                    range_set = false;
                }
                last_char = c;
            }
            break;
        case 'b' :
            if(escaped){
                column.set('\b',value);
                if(range_set){
                    setRange(column,range_start,'\b',value);
                    range_set = false;
                }
                last_char = '\b';
                escaped = false;
            }else{
                column.set(c, value);
                if(range_set){
                    //std::cout << "RANGE SET" << std::endl;
                    setRange(column,range_start,c,value);
                    range_set = false;
                }
                last_char = c;
            }
            break;
        case 'f' :
            if(escaped){
                column.set('\f',value);
                if(range_set){
                    setRange(column,range_start,'\f',value);
                    range_set = false;
                }
                last_char = '\f';
                escaped = false;
            }else{
                column.set(c, value);
                if(range_set){
                    setRange(column,range_start,c,value);
                    range_set = false;
                }
                last_char = c;
            }
            break;
        case 'v' :
            if(escaped){
                column.set('\v',value);
                if(range_set){
                    setRange(column,range_start,'\v',value);
                    range_set = false;
                }
                last_char = '\v';
                escaped = false;
            }else{
                column.set(c, value);
                if(range_set){
                    setRange(column,range_start,c,value);
                    range_set = false;
                }
                last_char = c;
            }
            break;
        case '\'' :
            if(escaped){
                column.set('\'',value);
                if(range_set){
                    setRange(column,range_start,'\'',value);
                    range_set = false;
                }
                last_char = '\'';
                escaped = false;
            }else{
                column.set(c, value);
                if(range_set){
                    setRange(column,range_start,c,value);
                    range_set = false;
                }
                last_char = c;
            }
            break;
        case '\"' :
            if(escaped){
                column.set('\"',value);
                if(range_set){
                    setRange(column,range_start,'\"',value);
                    range_set = false;
                }
                last_char = '\"';
                escaped = false;
            }else{
                column.set(c, value);
                if(range_set){
                    setRange(column,range_start,c,value);
                    range_set = false;
                }
                last_char = c;
            }
            break;
            /*
        case '?' :
            if(escaped){
                column.set('?',value);
                last_char = '?';
                escaped = false;
            }else{
                column.set(c, value);
                last_char = c;
            }
            break;
            */
            // Range
        case '-' :
            // only set the range if the previous char wasn't a bracket
            if(escaped || last_char == OPEN_BRACKET){
                column.set('-',value);
                if(range_set){
                    setRange(column,range_start,'-',value);
                    range_set = false;
                }
                escaped = false;
                last_char = '-';
            }else{
                range_set = true;
                range_start = last_char;
            }
            break;

            // Special Classes
        case 's' :
            if(escaped){
                column.set('\n',value);
                column.set('\t',value);
                column.set('\r',value);
                column.set('\x0B',value); //vertical tab
                column.set('\x0C',value);
                column.set('\x20',value);
                escaped = false;
            }else{
                column.set(c, value);
                if(range_set){
                    setRange(column,range_start,c,value);
                    range_set = false;
                }
                last_char = c;
            }
            break;

        case 'd' :
            if(escaped){
                setRange(column,48,57, value);
                escaped = false;
            }else{
                column.set(c, value);
                if(range_set){
                    setRange(column,range_start,c,value);
                    range_set = false;
                }
                last_char = c;
            }
            break;

        case 'w' :
            if(escaped){
                column.set('_', value); // '_'
                setRange(column,48,57, value); // d
                setRange(column,65,90, value); // A-Z
                setRange(column,97,122, value); // a-z
                escaped = false;
            }else{
                column.set(c, value);
                if(range_set){
                    setRange(column,range_start,c,value);
                    range_set = false;
                }
                last_char = c;
            }
            break;

            // Inversion
        case '^' :
            if(escaped){
                column.set(c,value);
                if(range_set){
                    setRange(column,range_start,c,value);
                    range_set = false;
                }
                last_char = c;
                escaped = false;
            }else{
                inverting = true;
            }
            break;

            // HEX
        case 'x' :
            if(escaped){
                //process hex char
                ++index;
                char hex[3];
                hex[0] = (char)symbol_set.c_str()[index];
                hex[1] = (char)symbol_set.c_str()[index+1];
                hex[2] = '\0';
                unsigned char number = (unsigned char)std::strtoul(hex, NULL, 16);
                
                //
                ++index;
                column.set(number, value);
                if(range_set){
                    setRange(column,range_start,number,value);
                    range_set = false;
                }
                last_char = number;
                escaped = false;
            }else{
                column.set(c, value);
                if(range_set){
                    setRange(column,range_start,c,value);
                    range_set = false;
                }
                last_char = c;
            }
            break;


            // Other characters
        default:
            if(escaped){
                // we escaped a char that is not valid so treat it normaly
                escaped = false;
            }
            column.set(c, value);
            if(range_set){
                setRange(column,range_start,c,value);
                range_set = false;
            }
            last_char = c;
        };

        index++;
    } // char while loop

    if(inverting)
        column.flip();

    if(bracket_sem != 0 || 
       brace_sem != 0){
        std::cout << "MALFORMED BRACKETS OR BRACES: " << symbol_set <<  std::endl;
        std::cout << "brackets: " << bracket_sem << std::endl;
        exit(1);
    }

}


