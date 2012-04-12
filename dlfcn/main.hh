

template<class TYPE>
TYPE function_cast(void * symbol)
{
   assert(sizeof(void *) == sizeof(TYPE));
   union
	{
	   void * symbol;
	   TYPE function;
	} cast;
   cast.symbol = symbol;
   return cast.function;
}


