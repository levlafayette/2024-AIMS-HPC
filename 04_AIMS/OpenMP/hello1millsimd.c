#include <stdio.h> 
#include "omp.h"
int main(void) 
{ 
   char greetings[] = "Hello world!"; 
   int a;
   #pragma omp parallel for simd
   for ( a = 0; a < 1000000; a = a + 1 )
   	{
	printf("%s\n", greetings); 
   	}
return 0;
}
