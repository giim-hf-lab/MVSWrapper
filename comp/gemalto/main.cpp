#ifdef __INTELLISENSE__

#include "gemalto/legacy/superdog.hpp"

#else

import gemalto;

#endif

int main()
{
	gemalto::superdog superdog;
	return not superdog.open() or not superdog.check_time();
}
