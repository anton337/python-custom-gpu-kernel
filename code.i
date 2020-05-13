%module code
%{
#include "code.h"
%}
%include "std_vector.i"
namespace std {
  %template(VecFloat) vector<float>;
  %template(VecVecfloat) vector<vector<float> >;
}

%include "code.h"
