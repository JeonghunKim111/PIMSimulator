#include "tests/csc/CSCMatrixLoader.h"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
namespace csc_descriptor {
CSCMatrix makeCSC(uint32_t rows,uint32_t cols,std::vector<COOEntry> e){
 std::stable_sort(e.begin(),e.end(),[](auto&a,auto&b){return a.col!=b.col?a.col<b.col:a.row<b.row;});
 CSCMatrix m;m.rows=rows;m.cols=cols;m.col_ptr.assign(size_t(cols)+1,0);
 for(auto&x:e){if(x.row>=rows||x.col>=cols)throw std::runtime_error("entry exceeds dimensions");m.col_ptr[x.col+1]++;}
 for(size_t i=1;i<m.col_ptr.size();++i)m.col_ptr[i]+=m.col_ptr[i-1];
 for(auto&x:e){m.row_idx.push_back(x.row);m.values.push_back(x.value);}return m;
}
static CSCMatrix loadMM(std::ifstream&in){
 std::string l,magic,obj,fmt,field,sym;std::getline(in,l);std::istringstream(l)>>magic>>obj>>fmt>>field>>sym;
 if(magic!="%%MatrixMarket"||obj!="matrix"||fmt!="coordinate"||field=="complex")throw std::runtime_error("unsupported Matrix Market input");
 do{if(!std::getline(in,l))throw std::runtime_error("missing dimensions");}while(l.empty()||l[0]=='%');
 uint64_t nr,nc,nz;std::istringstream(l)>>nr>>nc>>nz;if(nr>UINT32_MAX||nc>UINT32_MAX)throw std::runtime_error("dimensions exceed uint32");
 bool pattern=field=="pattern",symmetric=sym=="symmetric"||sym=="hermitian",skew=sym=="skew-symmetric";std::vector<COOEntry> e;e.reserve(nz*(symmetric||skew?2:1));
 uint64_t r,c;double v;while(in>>r>>c){v=1;if(!pattern)in>>v;if(!r||!c)throw std::runtime_error("MM indices must be one-based");--r;--c;e.push_back({uint32_t(r),uint32_t(c),float(v)});if((symmetric||skew)&&r!=c)e.push_back({uint32_t(c),uint32_t(r),float(skew?-v:v)});}return makeCSC(nr,nc,std::move(e));
}
static CSCMatrix loadTrip(std::ifstream&in){
 std::vector<std::string> ls;std::string l;while(std::getline(in,l))if(!l.empty()&&l[0]!='#'&&l[0]!='%')ls.push_back(l);if(ls.empty())return makeCSC(0,0,{});
 uint64_t a,b,c;std::istringstream f(ls[0]);bool hdr=bool(f>>a>>b>>c)&&f.eof()&&ls.size()==size_t(c)+1;uint64_t nr=hdr?a:0,nc=hdr?b:0;std::vector<COOEntry> e;
 for(size_t i=hdr?1:0;i<ls.size();++i){uint64_t r,col;float v=1;std::istringstream s(ls[i]);if(!(s>>r>>col))continue;s>>v;if(r>UINT32_MAX||col>UINT32_MAX)throw std::runtime_error("index exceeds uint32");e.push_back({uint32_t(r),uint32_t(col),v});nr=std::max(nr,r+1);nc=std::max(nc,col+1);}return makeCSC(nr,nc,std::move(e));
}
CSCMatrix loadCSC(const std::string&p){std::ifstream in(p);if(!in)throw std::runtime_error("failed to open "+p);std::string f;std::getline(in,f);in.clear();in.seekg(0);return f.rfind("%%MatrixMarket",0)==0?loadMM(in):loadTrip(in);}
}
