#ifndef INST_H
#define INST_H
#include <nvToolsExt.h>

extern const uint32_t colors[];
extern const int num_colors;
#define PUSH_RANGE(name,cid) { \
int color_id = cid; \
color_id = color_id%num_colors;\
nvtxEventAttributes_t eventAttrib = {0}; \
eventAttrib.version = NVTX_VERSION; \
eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
eventAttrib.colorType = NVTX_COLOR_ARGB; \
eventAttrib.color = colors[color_id]; \
eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
eventAttrib.message.ascii = name; \
nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name,cid)
#define POP_RANGE

#endif
