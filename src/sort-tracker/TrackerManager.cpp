

/*
    Module: Object Tracking
    Author: Le Manh Thang
    Created: August 26th, 2021
*/
#include "TrackerManager.hpp"
#include <iostream>
#include <fstream>
#include <iomanip> // to format image names using setw() and setfill()
//#include <io.h>    // to check file existence using POSIX function access(). On Linux 
#include <unistd.h>
#include <set>


TrackerManager::TrackerManager(Rect2f rect)
{
    this->tracker = KalmanTracker(rect);
}

TrackerManager::~TrackerManager()
{

}

void TrackerManager::update(Rect2f rect)
{   
    this->tracker.update(rect);
    this->tracer.m_rect = rect;
    this->tracer.m_ID = this->tracker.m_id + 1;
    this->tracer.m_trace.push_back(this->tracker.get_point());
    if (this->tracer.m_trace.size() > max_trace_length)
    {
        this->tracer.m_trace.pop_front(this->tracer.m_trace.size() - max_trace_length);
    }
}

TrackingTrace TrackerManager::get() const
{   
    return this->tracer;
}

Rect2f TrackerManager::predict()
{
    return this->tracker.predict();
}

bool TrackerManager::isConfirmed()
{
    return ((this->tracker.m_time_since_update < 1) &&
            (this->tracker.m_hit_streak >=  min_confirmed));
}

bool TrackerManager::isDeleted()
{
    return (this->tracker.m_time_since_update > max_age);
}

bool TrackerManager::isTentative()
{
    return (this->tracker.m_time_since_update > min_confirmed);
}
