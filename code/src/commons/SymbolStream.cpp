/*
 * SymbolStream.cpp
 *
 *  Created on: May 1, 2018
 *      Author: hyliu
 */

#include "SymbolStream.h"

#include <string>
#include <set>
#include <vector>
#include <iostream>
#include <fstream>
#include <iterator>
#include <sstream>
#include <cassert>
#include <cstdlib>
#include <time.h>

using std::string;
using std::set;
using std::vector;
using std::endl;
using std::cout;
using std::ios;


SymbolStream::SymbolStream() {
}

SymbolStream::~SymbolStream() {
}


const set<uint8_t>&  SymbolStream::calc_alphabet() {
	this->alphabet.clear();
	for (int i = 0; i < input.size(); i++) {
		alphabet.insert(input[i]);
	}

	//cout << "size_of_alphabet = " << alphabet.size() << endl;
	return alphabet;
}


int SymbolStream::get_length() const {
    return input.size();
}


/**
 * From VASim
 */
static vector<unsigned char> file2CharVector(string fn) {

    // open the file:
    std::ifstream file(fn, std::ios::binary);
    if(file.fail()){
        if(errno == ENOENT) {
            cout<< " Error: no such input file." << endl;
            exit(-1);
        }
    }

    // get its size:
    std::streampos fileSize;

    file.seekg(0, std::ios::end);
    fileSize = file.tellg();
    file.seekg(0, ios::beg);

    // Stop eating new lines in binary mode!!!
    file.unsetf(std::ios::skipws);

    // reserve capacity
    std::vector<unsigned char> vec;
    vec.reserve(fileSize);

    // read the data:
    vec.insert(vec.begin(),
               std::istream_iterator<unsigned char>(file),
               std::istream_iterator<unsigned char>());

    return vec;

}

static vector<unsigned char> generateRandomCharVector() {
  std::string path = "./random.txt";
  std::ofstream randomFile(path);
  std::vector<unsigned char> vec;
  std::srand(time(0));

  printf("Save random file to %s\n", path.c_str());
  if (randomFile.is_open()) {
    for (int i = 0; i < 1000050; i++) {
      char c = rand() % 256;
      vec.push_back(c);
      randomFile << c;
    }
    randomFile.close();
  }
  return vec;
}

void SymbolStream::readFromFile(string filename) {
    string input_fn = filename;
    bool randomInput = false;
    vector<unsigned char> input2;

    auto hasEnding = [](std::string const &fullString,
                        std::string const &ending) -> bool {
      if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare(fullString.length() - ending.length(),
                                        ending.length(), ending));
      } else {
        return false;
      }
    };

    if (hasEnding(input_fn, "random")) {
      randomInput = true;
      std::cout << "generate random input\n";
      input2 = generateRandomCharVector();
    } else {
      cout << "read input stream from file = " << input_fn << endl;
      input2 = file2CharVector(input_fn);
    }

    input.clear();

    // copy bytes to unsigned ints
    uint32_t counter = 0;

    for(uint8_t val : input2){
    	input.push_back(val);
    }

    cout << "input_stream_size = " << input.size() << endl;

    this->fromFile = filename;
}


uint8_t SymbolStream::get_position(int pos) const {
	return this->input[pos];
}

void SymbolStream::set_position(int pos, uint8_t c) {
    assert(pos >= 0 && pos < size());
    this->input[pos] = c;
}

SymbolStream SymbolStream::slice(int start, int len) const {
    SymbolStream res;
    
    assert(start >= 0);
    assert(len >= 0);
    assert(start < this->input.size());

    if (start + len > this->input.size()) {
        len = this->input.size() - start;
        cout << "the input is shorter than the length specified, just slice to end" << endl;
    }

    assert(start + len <= this->input.size());
    
    for (int i = start; i < start + len; i++) {
        res.input.push_back(this->input[i]);
    }

    return res;
}




void SymbolStream::padding_to_base(int base) {
    if (base <= 0) {
        return;
    }

    while (this->size() % base != 0) {
        this->input.push_back( (uint8_t) 0);
    }
}