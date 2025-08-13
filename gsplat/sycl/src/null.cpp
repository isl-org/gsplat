 
#include <c10/xpu/XPUStream.h>

#include "Ops.h"
#include "Common.h"

namespace  gsplat::xpu {

at::Tensor null(const at::Tensor input) {
    throw std::runtime_error(std::string(__func__) + " is not implemented");
}

} //namespace  gsplat::xpu