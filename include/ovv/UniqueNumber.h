#ifndef UNIQUE_NUMBER
#define UNIQUE_NUMBER

// class generator:
class UniqueNumber 
{
    public:
        UniqueNumber() {current=0;};
        int operator()() {return current++;};
    private:
        int current;
};

#endif
