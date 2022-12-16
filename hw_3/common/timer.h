#ifndef TIMER_H
#define TIMER_H

#include <stdio.h>
#include <sys/time.h>

class Timer
{
private:
    double time;

    double getTimeInNs()
    {
        struct timeval tp;
        gettimeofday(&tp, NULL);
        return ((double)tp.tv_sec * 1e6 + (double)tp.tv_usec);
    }

public:
    Timer(){};
    void start()
    {
        time = getTimeInNs();
    }
    void stop(const char *operation)
    {
        double timePassed = getTimeInNs() - time;
        printf("Operation %s took %.1fµs to complete.\n", operation, timePassed);
    }
};

#endif