
#include "common_func.h"

#include "NFA.h"
#include <unordered_map>
#include <iostream>
#include <string>
#include <algorithm>
#include <cassert>
#include <queue>
#include <stack>
#include <memory>
#include <map>
#include <vector>
#include <iomanip>
#include <fstream>
#include <climits>
#include "nfa_utils.h"

#include <sys/types.h>
#include <sys/stat.h>



using std::ifstream;
using std::string;
using std::endl;
using std::cout;
using std::pair;





void tools::create_path_if_not_exists(string path) {
		struct stat info;

		if( stat( path.c_str(), &info ) != 0 )
		{
		    printf( "cannot access %s\n", path.c_str() );

		    const int dir_err = mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
			if (-1 == dir_err)
			{
    			printf("Error creating directory!n");
    			exit(1);
			} else {
				puts("success create dir ");
			}

		}

		else if( info.st_mode & S_IFDIR )  // S_ISDIR() doesn't exist on my windows  
	    {
	    	printf( "%s is a directory\n", path.c_str() );
	    		
	    }
		else
	    {
	    	printf( "%s is no directory\n", path.c_str() );
	    	exit(-1);
	    }

}