
#include <cmath>
#include <iostream>
#include <vector>
#include <map>

long double f3_formula(const long double& r)
{
   long double result = 0.0;
   if(r <= 3.0){ result += 3.0 - r; }
   if(r <= 1.0){ result -= 3.0*(1.0-r); }
   return result;
}

long double f4_formula(const long double& r)
{
   long double x = (2.0-r)*(2.0-r)*0.5;
   long double y = r*r*0.5*3.0;
   long double result = 0.0;
   if(r <= 4.0){ result += x; }
   if(r <= 2.0){ result += y-x ; }
   return result;
}


long double fn_integral(const long double& r, const int& order)
{
   static const int nb_points = 2500*2;
   long double lower = std::fabs(r-1.0);
   long double upper = r+1.0;
   long double dr = (upper-lower)/(nb_points-1);
   long double accum = 0.0;
   if(order==4)
   {
      //for(int i=0;i<nb_points;i++)
      //{
      //   long double rp = lower + dr*i;
      //   accum += dr*f3_formula(rp);
      //}
      accum = f4_formula(r);
   }
   else
   {
      for(int i=0;i<nb_points;i++)
      {
         long double rp = lower + dr*i;
         accum += dr*fn_integral(rp,order-1);
      }
   }
   return accum;
}


class FN
{
  public:
    FN(int samples, int orders, long double rmn, long double rmx ):
      nb_samples(samples),
      nb_orders(orders),
      rmin(rmn),
      rmax(rmx)
    {
       init();
    }


    long double eval( int order, long double r ) const ;

  private:

    int nb_samples;
    int nb_orders;
    long double rmin;
    long double rmax;

    std::map<int, std::vector<long double> > fNsets;
    void init();
};

void FN::init()
{
   // order 3
   std::vector<long double> f3(nb_samples); 
   long double dr = (rmax-rmin)/(nb_samples-1);
   for(int i=0;i<nb_samples;i++)
   {
      long double r = rmin + i*dr;
      f3[i] = f3_formula(r);
   }
   fNsets[3] = f3;
   std::vector<long double> f4(nb_samples); 
   for(int i=0;i<nb_samples;i++)
   {
      long double r = rmin + i*dr;
      f4[i] = f4_formula(r);
   }
   fNsets[4] = f4;

   for(int o=5;o<=nb_orders;o++)
   {
      std::cout << "# order=" << o << std::endl;
      std::vector<long double> f(nb_samples); 
#pragma omp parallel for
      for(int i=0;i<nb_samples;i++)
      {
         long double r = rmin + i*dr;
         long double lower = std::fabs(r-1.0);
         long double upper = r+1.0;
         long double ddr = (upper-lower)/(nb_samples-1);
         long double accum = 0.0;
         //const std::vector<long double>& previous = fNsets.at(o-1);
         for(int i=0;i<nb_samples;i++)
         {
               long double rp = lower + i*ddr;
	       /*
	       long double rrr = (rp-rmin)/dr;
	       int ii = rrr;
	       long double weight = rrr - ii;
               if(ii < 0 ){ ii = 0; }
	       if(ii >= nb_samples){ii = nb_samples; }
	       int iii = ii + 1;
               if(iii < 0 ){ iii = 0; }
	       if(iii >= nb_samples){iii = nb_samples; }
               accum += ddr*(previous[ii] * (1.0-weight) + previous[iii]*weight);
	       */
	       accum += ddr*eval(o-1,rp);
         }
	 f[i] = accum;
      }
      fNsets[o] = f;
   }
}

long double FN::eval( int order, long double r ) const
{
   const std::vector<long double>& data = fNsets.at(order);
   long double rrr = (r-rmin)*(nb_samples-1)/(rmax-rmin);
   int ii = rrr;
   long double weight = rrr - ii;
   if(ii < 0 ){ ii = 0; }
   if(ii >= nb_samples){ii = nb_samples; }
   int iii = ii + 1;
   if(iii < 0 ){ iii = 0; }
   if(iii >= nb_samples){iii = nb_samples; }
   return data[ii] * (1.0-weight) + data[iii]*weight; 
}


void compute(int order)
{
   long double z = std::fabs( 0.2*order - 2.0 );
   long double rmin = std::fabs(z-1.0);
   long double rmax = std::fabs(z+1.0);
   long double dr = (rmax-rmin)/100.0;
   long double r = rmin;
   std::cout << "# Order=" << order << std::endl << std::flush;
   while(r<rmax+dr)
   {
      long double f = (order==3) ? f3_formula(r) : fn_integral(r,order);
      long double s = (r*r - z*z - 1.0)/(2.0*z);
      std::cout << s << " " << r << " " << f << std::endl << std::flush;
      r += dr;
   }
}

void compute(int order, FN& fn)
{
   long double z = std::fabs( 0.21234*order - 2.0 );
   long double rmin = std::fabs(z-1.0);
   long double rmax = std::fabs(z+1.0);
   long double dr = (rmax-rmin)/100.0;
   long double r = rmin;
   long double mag = std::pow(2.0*3.14159265,order-1);
   long double r0 = std::sqrt(z*z+1.0);
   long double fo = fn.eval(order,r0);
   long double pdfo = fo * mag;
   long double compare_pdfo = std::pow( 2.0*3.14159265*std::exp(0.625),order-1 )*std::exp(-0.375);
   long double lnfo = std::log(fo);
   long double compare_lnfo = 0.625*order - 1.0; 
   std::cout << "# Order=" << order << "  " << pdfo << " " << compare_pdfo << " " << lnfo << " " << compare_lnfo << std::endl << std::flush;
   while(r<rmax+dr)
   {
      long double f = fn.eval(order,r);
      long double s = (r*r - z*z - 1.0)/(2.0*z);
      long double pdf = f*mag/r;
      std::cout << s << " " << r << " " << f << "  " << pdf << std::endl << std::flush;
      r += dr;
   }
}

void compute_bounds(int order, FN& fn)
{
   long double xa = 0;
   long double xxa = 0;
   long double ya = 0;
   long double xya = 0;
   int count = 0;
   for(int o=3;o<=order;o++)
   {
      long double z = std::fabs( 0.21234*o - 2.0 );
      long double rmin = std::fabs(z-1.0);
      long double fmin = fn.eval(o,rmin);
      if( o>6 )
      {
         xa += o;
         xxa += o*o;
         ya += std::log(fmin);
         xya += std::log(fmin)*o;
         count++;
      }
      std::cout << o << " " << fmin << std::endl << std::flush;
   }
   xa /= count;
   xxa /= count;
   ya /= count;
   xya /= count;

   long double lnB = ( xya*xa - xxa*ya )/( xa*xa - xxa );
   long double A = ( ya - lnB )/xa;
   std::cout << " fmin ~ exp(" << A << " * order + " << lnB <<" )\n";

   std::cout << "\n\n\n\n";
   xa = 0;
   xxa = 0;
   ya = 0;
   xya = 0;
   count = 0;
   for(int o=3;o<=order;o++)
   {
      long double z = std::fabs( 0.21234*o - 2.0 );
      long double rmax = std::fabs(z+1.0);
      long double fmax = fn.eval(o,rmax);
      if( o>6 )
      {
         xa += o;
         xxa += o*o;
         ya += std::log(fmax);
         xya += std::log(fmax)*o;
         count++;
      }
      std::cout << o << " " << fmax << std::endl << std::flush;
   }
   xa /= count;
   xxa /= count;
   ya /= count;
   xya /= count;

   lnB = ( xya*xa - xxa*ya )/( xa*xa - xxa );
   A = ( ya - lnB )/xa;
   std::cout << " fmax ~ exp(" << A << " * order + " << lnB <<" )\n";


}




int main()
{
   FN fn(10000, 200, 0.0, 200.0 );
   for(int order=3;order<201;order++)
   {
       //compute(order);
       compute(order,fn);
      //compute_bounds(order,fn);
      std::cout << "\n\n\n\n" << std::flush;
   }

   return 0;
}
