// Fixed precision rational tridiagonal kernel.
// Basic real operations use Boost cpp_bin_float; complex operations are expanded.
#include <boost/version.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <vector>
#include <cstddef>
#include <limits>
using Index = std::ptrdiff_t;
std::size_t checked_count(Index a, Index b){
 if(a<0 || b<0) throw std::length_error("negative allocation extent");
 auto x=static_cast<std::size_t>(a), y=static_cast<std::size_t>(b);
 if(y && x>static_cast<std::size_t>(std::numeric_limits<Index>::max())/y)
  throw std::length_error("allocation extent exceeds addressable index range");
 return x*y;
}
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#ifndef WORK_BITS
#define WORK_BITS 512
#endif
#ifndef TWOSIDED_SUPPLY
#define TWOSIDED_SUPPLY 1
#endif
#ifndef OUTWARD_GAUGE
#define OUTWARD_GAUGE 1
#endif
#ifndef BALANCED_GAUGE
#define BALANCED_GAUGE 1
#endif
using R=boost::multiprecision::number<boost::multiprecision::cpp_bin_float<WORK_BITS,boost::multiprecision::digit_base_2>>;
struct Z{
 R re,im;
 Z():re(0),im(0){} Z(Index a):re(a),im(0){} Z(R a):re(a),im(0){} Z(R a,R b):re(a),im(b){}
 Z operator+(const Z&b)const{return {re+b.re,im+b.im};}
 Z operator-(const Z&b)const{return {re-b.re,im-b.im};}
 Z operator-()const{return {-re,-im};}
 Z operator*(const Z&b)const{return {re*b.re-im*b.im,re*b.im+im*b.re};}
 Z operator/(const Z&b)const{R d=b.re*b.re+b.im*b.im;if(d==0)throw std::runtime_error("zero complex divisor");return {(re*b.re+im*b.im)/d,(im*b.re-re*b.im)/d};}
 Z&operator+=(const Z&b){re+=b.re;im+=b.im;return *this;}
 Z&operator-=(const Z&b){re-=b.re;im-=b.im;return *this;}
};
using V=std::vector<Z>;
R az(const Z&z){return sqrt(z.re*z.re+z.im*z.im);}
R norm_squared(const Z&z){return z.re*z.re+z.im*z.im;}
Z dot(const Z*a,const Z*b,Index r){Z s;for(Index j=0;j<r;++j)s+=a[j]*b[j];return s;}
Z dot(const V&a,const V&b){return dot(a.data(),b.data(),(Index)a.size());}
V mv(const V&M,const Z*x,Index r){V y(r);for(Index i=0;i<r;++i)y[i]=dot(M.data()+i*r,x,r);return y;}
std::string rs(R x){return x.str(100,std::ios_base::scientific);}
void jnum(std::ostream&o,R x){o<<'"'<<rs(x)<<'"';}
struct Kernel{
 Index n,r;V a,U,Vgen,P;std::vector<bool> pin;std::vector<R> td,te;
 boost::multiprecision::cpp_int state_entries=0,matvec_terms=0,linear_terms=0,resolvent_rows=0;
 R max_generator=0,max_transition=0,min_resolvent_pivot=1;
 R balanced_max_transition=0,max_scale=1,min_scale=1;
 R minimum_damping_ratio=2,minimum_twist_damping_ratio=2;
 boost::multiprecision::cpp_int supply_divisions=0;
 R state_peak=0,state_derivative_peak=0,backward_peak=0,backward_derivative_peak=0;
 // Diagnostic maxima accumulate squared magnitudes. A monotone square root
 // is needed only when exporting the maximum, never for every visited entry.
 void peak(const V&M,R&v){for(const auto&x:M)v=std::max(v,norm_squared(x));}
 Kernel(Index nn,Index rr,std::vector<R>d,std::vector<R>e,R c,V poles,V weights):n(nn),r(rr),a(n,Z(c)),U(checked_count(n,r)),Vgen(checked_count(n,r),Z(1)),P(checked_count(n,r),Z(1)),pin(n,false),td(d),te(e){
  for(Index j=0;j<r;++j){V piv(n),g(n);
   for(Index i=0;i<n;++i){piv[i]=Z(d[i])-poles[j];if(i)piv[i]-=Z(e[i-1]*e[i-1])/piv[i-1];
    R ap=norm_squared(piv[i]);if(ap==0)throw std::runtime_error("zero resolvent pivot");if(ap<min_resolvent_pivot)min_resolvent_pivot=ap;++resolvent_rows;}
   for(Index i=0;i<n;++i){
     R dr=-piv[i].im/poles[j].im;
     minimum_damping_ratio=std::min(minimum_damping_ratio,dr);
   }
#if TWOSIDED_SUPPLY
   // Independent right pivots, then the Schur-complement diagonal of the inverse.
   // No summation of t_i^2*g_(i+1) along the backward chain.
   V right(n);
   for(Index i=n-1;i>=0;--i){
     right[i]=Z(d[i])-poles[j];
     if(i+1<n){right[i]-=Z(e[i]*e[i])/right[i+1];++supply_divisions;}
     minimum_damping_ratio=std::min(minimum_damping_ratio,R(-right[i].im/poles[j].im));
     ++resolvent_rows;
   }
   for(Index i=0;i<n;++i){
     Z twist=piv[i];
     if(i+1<n){twist-=Z(e[i]*e[i])/right[i+1];++supply_divisions;}
     minimum_twist_damping_ratio=std::min(minimum_twist_damping_ratio,R(-twist.im/poles[j].im));
     g[i]=Z(1)/twist;++supply_divisions;++resolvent_rows;
   }
#else
   g[n-1]=Z(1)/piv[n-1];
   for(Index i=n-2;i>=0;--i){Z t=-Z(e[i])/piv[i];g[i]=Z(1)/piv[i]+t*t*g[i+1];++resolvent_rows;}
#endif
   for(Index i=0;i<n;++i){U[i*r+j]=weights[j]*g[i];a[i]+=U[i*r+j];if(i)P[i*r+j]=-Z(e[i-1])/piv[i-1];
    max_generator=std::max(max_generator,norm_squared(U[i*r+j]));max_transition=std::max(max_transition,norm_squared(P[i*r+j]));}
  }
  // Energy gauge: s_i^2 = 1 + |phi_i|^2 s_(i-1)^2.
  // U_i <- U_i*s_i; V_i <- V_i/s_i;
  // P_i <- P_i*s_(i-1)/s_i.  Exact-arithmetic matrix is unchanged.
  for(Index j=0;j<r;++j){R prev=1;
    for(Index i=0;i<n;++i){R sc=1;
#if BALANCED_GAUGE
      if(i){R t=az(P[i*r+j])*prev;sc=sqrt(1+t*t);
#if OUTWARD_GAUGE
        // Strict contraction survives rounding. The computed scale cancels
        // algebraically; it need not approximate any particular ideal gauge.
        sc*=R(1)+ldexp(R(1),7-WORK_BITS);
#endif
        P[i*r+j]=P[i*r+j]*Z(prev/sc);}
      U[i*r+j]=U[i*r+j]*Z(sc);Vgen[i*r+j]=Z(1/sc);
#endif
      max_scale=std::max(max_scale,sc);min_scale=std::min(min_scale,sc);
      balanced_max_transition=std::max(balanced_max_transition,norm_squared(P[i*r+j]));prev=sc;
    }
  }
  max_generator=sqrt(max_generator);max_transition=sqrt(max_transition);
  balanced_max_transition=sqrt(balanced_max_transition);
  min_resolvent_pivot=sqrt(min_resolvent_pivot);
 }
 struct Factors{V d,dd,b,bd,w,wd,rd,hd;Factors(Index n,Index r):d(n),dd(n),b(checked_count(n,r),Z(1)),bd(checked_count(n,r)),w(checked_count(n,r)),wd(checked_count(n,r)),rd(n),hd(n){} };
 Factors scores(){
  Factors f(n,r);V S(checked_count(r,r)),Sd(checked_count(r,r));bool active=false;
  for(Index i=0;i<n;++i){const Z*u=&U[i*r];const Z*p=&P[i*r];
   if(active){
    if(i)for(Index j=0;j<r;++j)for(Index k=j;k<r;++k){Z pp=p[j]*p[k];S[j*r+k]=S[j*r+k]*pp;Sd[j*r+k]=Sd[j*r+k]*pp;S[k*r+j]=S[j*r+k];Sd[k*r+j]=Sd[j*r+k];state_entries+=2;}
    V su=mv(S,u,r),sdu=mv(Sd,u,r);matvec_terms+=2ULL*r*r;
    f.d[i]=a[i]-dot(u,su.data(),r);f.dd[i]=Z(pin[i]?1:0)-dot(u,sdu.data(),r);
    for(Index j=0;j<r;++j){f.b[i*r+j]=Vgen[i*r+j]-su[j];f.bd[i*r+j]=-sdu[j];}
   }else{f.d[i]=a[i];f.dd[i]=Z(pin[i]?1:0);for(Index j=0;j<r;++j)f.b[i*r+j]=Vgen[i*r+j];}
   if(pin[i]){
    if(f.d[i].re<=0)throw std::runtime_error("nonpositive pin factor");
    Z dinv=Z(1)/f.d[i],dinvd=-f.dd[i]*dinv*dinv;
    for(Index j=0;j<r;++j){f.w[i*r+j]=f.b[i*r+j]*dinv;f.wd[i*r+j]=f.bd[i*r+j]*dinv+f.b[i*r+j]*dinvd;}
    for(Index j=0;j<r;++j)for(Index k=j;k<r;++k){Z bprod=f.b[i*r+j]*f.b[i*r+k];
     S[j*r+k]+=bprod*dinv;
     Sd[j*r+k]+=(f.bd[i*r+j]*f.b[i*r+k]+f.b[i*r+j]*f.bd[i*r+k])*dinv+bprod*dinvd;
     S[k*r+j]=S[j*r+k];Sd[k*r+j]=Sd[j*r+k];state_entries+=2;}
    active=true;
   }
   peak(S,state_peak);peak(Sd,state_derivative_peak);
  }
  // Release two quadratic arrays before allocating backward messages.
  S.clear();Sd.clear();S.shrink_to_fit();Sd.shrink_to_fit();
  V B(checked_count(r,r)),Bd(checked_count(r,r));active=false;
  for(Index i=n-1;i>=0;--i){const Z*u=&U[i*r];
   if(!pin[i]){
    if(active){V y=mv(B,&f.b[i*r],r),yd=mv(Bd,&f.b[i*r],r);matvec_terms+=2ULL*r*r;
     f.rd[i]=f.d[i]-dot(&f.b[i*r],y.data(),r);
     f.hd[i]=f.dd[i]-Z(2)*dot(&f.bd[i*r],y.data(),r)-dot(&f.b[i*r],yd.data(),r);
    }else{f.rd[i]=f.d[i];f.hd[i]=f.dd[i];}
   }else{
    V y(r),yd(r);
    if(active){y=mv(B,&f.w[i*r],r);yd=mv(Bd,&f.w[i*r],r);V extra=mv(B,&f.wd[i*r],r);for(Index j=0;j<r;++j)yd[j]+=extra[j];matvec_terms+=3ULL*r*r;}
    Z s=dot(&f.w[i*r],y.data(),r)+Z(1)/f.d[i];
    Z sd=Z(2)*dot(&f.wd[i*r],y.data(),r)+dot(&f.w[i*r],yd.data(),r)-dot(&f.wd[i*r],y.data(),r)-f.dd[i]/(f.d[i]*f.d[i]);
    // yd=Bdot*w+B*wd, so w*yd-wd*y=w*Bdot*w.
    for(Index j=0;j<r;++j)for(Index k=j;k<r;++k){Z uu=u[j]*u[k];
     B[j*r+k]+=-u[j]*y[k]-y[j]*u[k]+uu*s;
     Bd[j*r+k]+=-u[j]*yd[k]-yd[j]*u[k]+uu*sd;
     B[k*r+j]=B[j*r+k];Bd[k*r+j]=Bd[j*r+k];state_entries+=2;}
    active=true;
   }
   peak(B,backward_peak);peak(Bd,backward_derivative_peak);
   if(i&&active){const Z*p=&P[i*r];for(Index j=0;j<r;++j)for(Index k=j;k<r;++k){Z pp=p[j]*p[k];B[j*r+k]=B[j*r+k]*pp;Bd[j*r+k]=Bd[j*r+k]*pp;B[k*r+j]=B[j*r+k];Bd[k*r+j]=Bd[j*r+k];state_entries+=2;}}
  }
  return f;
 }
 V matvec(const V&x){V y(n),f(r);
  for(Index i=0;i<n;++i){y[i]=a[i]*x[i];if(i)for(Index j=0;j<r;++j)f[j]=P[i*r+j]*(f[j]+Vgen[(i-1)*r+j]*x[i-1]);y[i]+=dot(&U[i*r],f.data(),r);linear_terms+=3ULL*r;}
  std::fill(f.begin(),f.end(),Z());
  for(Index i=n-1;i>=0;--i){if(i<n-1)for(Index j=0;j<r;++j)f[j]=P[(i+1)*r+j]*(f[j]+U[(i+1)*r+j]*x[i+1]);for(Index j=0;j<r;++j)y[i]+=Vgen[i*r+j]*f[j];linear_terms+=3ULL*r;}
  return y;
 }
 V solve_pins(const Factors&f,const V&b){V y(n),x(n),state(r);
  for(Index i=0;i<n;++i){if(i)for(Index j=0;j<r;++j)state[j]=P[i*r+j]*state[j];if(pin[i]){y[i]=b[i]-dot(&U[i*r],state.data(),r);for(Index j=0;j<r;++j)state[j]+=f.w[i*r+j]*y[i];}linear_terms+=3ULL*r;}
  std::fill(state.begin(),state.end(),Z());
  for(Index i=n-1;i>=0;--i){if(pin[i]){x[i]=y[i]/f.d[i]-dot(&f.w[i*r],state.data(),r);for(Index j=0;j<r;++j)state[j]+=U[i*r+j]*x[i];}if(i)for(Index j=0;j<r;++j)state[j]=P[i*r+j]*state[j];linear_terms+=3ULL*r;}
  return x;
 }
 void run(Index k,std::ostream&o){
  o<<"{\"precision_bits\":"<<WORK_BITS<<",\"balanced_gauge\":"<<BALANCED_GAUGE<<",\"n\":"<<n<<",\"rank\":"<<r<<",\"columns\":[";
  R trace=0;R tol=ldexp(R(1),-std::min(250,WORK_BITS/2));
  for(Index it=0;it<k;++it){auto t0=std::chrono::steady_clock::now();Factors f=scores();Index pivot=-1;R best=0,maximag=0,minh=0;
   for(Index i=0;i<n;++i)if(!pin[i]){maximag=std::max(maximag,abs(f.rd[i].im));maximag=std::max(maximag,abs(f.hd[i].im));minh=std::min(minh,f.hd[i].re);
    if(f.rd[i].re<=0)continue;if(f.hd[i].re < -tol)throw std::runtime_error("negative squared interpolation norm exceeds roundoff diagnostic");
    R score=(1+std::max(R(0),f.hd[i].re))/f.rd[i].re;if(pivot<0||score<best){best=score;pivot=i;}}
   if(pivot<0)throw std::runtime_error("no positive candidate diagonal");
   V e(n);e[pivot]=Z(1);V b=matvec(e);V x=solve_pins(f,b);V fx=matvec(x);V col(n);R cn=0,pinmax=0,imh=0,xnorm=0;
   for(Index i=0;i<n;++i){col[i]=b[i]-fx[i];cn+=col[i].re*col[i].re;imh=std::max(imh,abs(col[i].im));if(pin[i]){pinmax=std::max(pinmax,az(col[i]));xnorm+=x[i].re*x[i].re;}}
   cn=sqrt(cn);if(cn==0)throw std::runtime_error("zero conditioned column");
   R hcheck=abs(xnorm-f.hd[pivot].re),dcheck=abs(col[pivot].re-f.rd[pivot].re);trace+=best;
   if(it)o<<',';o<<"{\"pivot\":"<<pivot<<",\"score\":";jnum(o,best);o<<",\"trace_inverse_sum\":";jnum(o,trace);
   o<<",\"diagonal\":";jnum(o,f.rd[pivot].re);o<<",\"interpolation_norm_sq\":";jnum(o,f.hd[pivot].re);
   o<<",\"score_crosscheck_abs\":";jnum(o,hcheck);o<<",\"diagonal_crosscheck_abs\":";jnum(o,dcheck);
   o<<",\"pin_coefficients\":[";bool firstpin=true;for(Index j=0;j<n;++j)if(pin[j]){if(!firstpin)o<<',';firstpin=false;o<<"["<<j<<",";jnum(o,x[j].re);o<<"]";}o<<"]";
   o<<",\"raw_column_norm\":";jnum(o,cn);o<<",\"pinned_raw_max_abs\":";jnum(o,pinmax);
   o<<",\"maximum_imaginary_diagnostic\":";jnum(o,std::max(maximag,imh));o<<",\"minimum_h_diagnostic\":";jnum(o,minh);
   o<<",\"vector\":[";for(Index i=0;i<n;++i){if(i)o<<',';jnum(o,col[i].re/cn);}o<<"]";
   o<<",\"all_scores\":[";for(Index i=0;i<n;++i){if(i)o<<',';if(pin[i]||f.rd[i].re<=0)o<<"null";else jnum(o,(1+f.hd[i].re)/f.rd[i].re);}o<<"]";
   o<<",\"elapsed_seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count()<<"}";o.flush();
   pin[pivot]=true;
  }
  o<<"],\"work\":{\"state_entry_updates\":"<<state_entries<<",\"state_matvec_terms\":"<<matvec_terms<<",\"linear_terms\":"<<linear_terms<<",\"resolvent_rows\":"<<resolvent_rows<<",\"full_vector_reorthogonalizations\":0},\"maximum_generator\":";jnum(o,max_generator);o<<",\"maximum_transition\":";jnum(o,max_transition);o<<",\"minimum_resolvent_pivot\":";jnum(o,min_resolvent_pivot);
  o<<",\"balanced_max_transition\":";jnum(o,balanced_max_transition);
  o<<",\"maximum_scale\":";jnum(o,max_scale);
  o<<",\"minimum_scale\":";jnum(o,min_scale);
  o<<",\"forward_state_peak\":";jnum(o,sqrt(state_peak));
  o<<",\"forward_derivative_peak\":";jnum(o,sqrt(state_derivative_peak));
  o<<",\"backward_state_peak\":";jnum(o,sqrt(backward_peak));
  o<<",\"backward_derivative_peak\":";jnum(o,sqrt(backward_derivative_peak));
  o<<"}\n";
 }
};

// Coefficients are imported as exact integer * power-of-two numbers.
// Their significands have fewer bits than WORK_BITS; import is exact.
R exact_dyadic(std::istream& in){
 std::string m;int e;in>>m>>e;
 if(!in)throw std::runtime_error("truncated dyadic input");
 return ldexp(R(m),e);
}
int main(int argc,char**argv){try{
 if(argc==2 && std::string(argv[1])=="--build-info"){
  std::cout<<"{\"precision_bits\":"<<WORK_BITS<<",\"boost_version\":"<<BOOST_VERSION
   <<",\"index_bits\":"<<std::numeric_limits<Index>::digits
   <<",\"source_sha256\":\""<<RBSVD_SOURCE_SHA256<<"\"}\n";return 0;
 }
 if(argc!=3)throw std::runtime_error("usage: rational-bidiagonal-svd backend input.txt output.json");
 std::ifstream in(argv[1]);Index n=0,k=0,r=0;in>>n>>k>>r;
 if(!in)throw std::runtime_error("invalid dimension header");
 if(n<1||k<1||k>n||r<1||r>1048576)throw std::runtime_error("invalid dimensions");
 if(!in)throw std::runtime_error("cannot open input");
 std::vector<R>d(n),e(n-1);for(auto&x:d)x=exact_dyadic(in);for(auto&x:e)x=exact_dyadic(in);
 R constant=exact_dyadic(in);V poles(r),weights(r);
 for(int j=0;j<r;++j){R re=exact_dyadic(in),im=exact_dyadic(in);poles[j]=Z(re,im);
   re=exact_dyadic(in);im=exact_dyadic(in);weights[j]=Z(re,im);}
 Kernel K(n,r,d,e,constant,poles,weights);
 std::ofstream out(argv[2]);if(!out)throw std::runtime_error("cannot open output");K.run(k,out);return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
