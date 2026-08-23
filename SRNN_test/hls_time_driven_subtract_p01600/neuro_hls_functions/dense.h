#pragma once

// Compatibility header.  The backend formerly called "dense"/"parallel"
// is now named "time-driven".  New generated projects include time_driven.h
// directly; this wrapper keeps older projects and external includes working.
#include "time_driven.h"
