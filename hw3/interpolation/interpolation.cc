/* ---------------------------------------------------------------------
* This problem is there to familiarize yourself with the basic of
* meshes in deal.II. It consists of creating a mesh and reading a gmsh
* file. Also, we will work with the refinement of the mesh by certain
* criteria, for this you will need to loop over all cells.
* ---------------------------------------------------------------------
*/


// This is the Triangulation class that is in charge of the mesh in deal.II.
// You will see this class in every deal.II program. It is documented here:
// https://www.dealii.org/developer/doxygen/deal.II/classTriangulation.html
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_out.h>

// The grid generator has several functions to great standard meshes in
// different dimensions, e.g., hypercubes, hyperballs, hypershells, etc
#include <deal.II/grid/grid_generator.h>


// You should be familiar with the Function class form homework 1
#include <deal.II/base/function.h>

#include <deal.II/fe/mapping_q.h>


// This class has the implementation of scalar Lagrange finite elements that
// lead to the finite element space of piecewise polynomials of certain degree
// in each coordinate direction. It is documented here:
// https://www.dealii.org/developer/doxygen/deal.II/classFE__Q.html
// There are many types of elements in deal.II but this is the most common one.
#include <deal.II/fe/fe_q.h>


// There are many quadrature rules implemented in deal.II
// (https://www.dealii.org/developer/doxygen/deal.II/group__Quadrature.html)
// this one corresponds to the Gauss-Legendre family, commonly called Gauss
// quadrature.
#include <deal.II/base/quadrature_lib.h>


// The FEValues class is what connects the finite element (shape functions) and
// the quadrature rule. It is in charge of evaluating shape functions at the
// points defined by the equadrature formula when mapped to the real cell.
// https://www.dealii.org/developer/doxygen/deal.II/classFEValues.html
#include <deal.II/fe/fe_values.h>


// The DataOut class enables us to output results to various formats
#include <deal.II/numerics/data_out.h>
// The vector_tools give us a namespace to calculate interpolations
#include <deal.II/numerics/vector_tools.h>


#include <cmath>
#include <fstream>
#include <iostream>


using namespace dealii;


// Function that we wish to interpolate
// The function is already fully defined
template <int dim>
class SineFunction : public Function<dim>
{
public:
 virtual double
 value(const Point<dim> &p, const unsigned int = 0) const override;
};


template <int dim>
double
SineFunction<dim>::value(const Point<dim> &p, const unsigned int) const
{
 double x = p[0];
 return std::sin(3 * x);

}


/**
* @brief Class that stores the entire interpolation problem.
* Instantiating this class with different constructor parameter will allow you
* to change the interpolation order and the initial refinement.
*/
template <int dim>
class Interpolator
{
public:
 /**
  * @brief Interpolator The constructor of the Interpolator class
  * @param order The order of the Lagrange polynomial you wish to use
  * @param number_of_initial_refinement The number of initial refinement in the mesh
  */
 Interpolator(const unsigned int order,
              const unsigned int number_of_initial_refinement)
   : order(order)
   , fe(order)
   , number_of_initial_refinement(number_of_initial_refinement)
   , mapping(order)
   , dof_handler(tria)
 {}




 /**
  * @brief setup_dofs Sets up the triangulation by creating a 1D
  * grid and refining it enough time
  */
 void
 setup_triangulation();


 /**
  * @brief setup_dofs Sets up the degrees of freedom to store the
  * values at colocation points
  */
 void
 setup_dofs();


 /**
  * @brief interpolate Carries out the actual interpolation
  */


 void
 interpolate();


 /**
  * @brief calculate_l2_error Caclculates the L2 norm of the error
  * between the interpolant and the function being interpolated.
  */


 void
 calculate_l2_error();


 /**
  * @brief output_results Output the results in a data file
  * whose naming convention is result_order_number_of_refinement.dat
  * The output file has two columns x and Interpolation(x)
  */
 void
 output_results();




 /**
  * @brief run Runs the entire interpolation.
  */
 void
 run();


private:
 const unsigned int order;
 const unsigned int number_of_initial_refinement;


 Triangulation<dim> tria;
 FE_Q<dim>          fe;
 MappingQ<dim>      mapping;
 Vector<double>     data_vector;
 DoFHandler<dim>    dof_handler;
};


template <int dim>
void
Interpolator<dim>::setup_triangulation()
{
 //************************
 // TODO
 //************************
 //  Generate a hyper_cube triangulation and refine it
 //  number_of_initial_refinement times


 GridGenerator::hyper_cube(tria, 0., 10.);
 tria.refine_global(number_of_initial_refinement);

 GridOut grid_out;
 std::ofstream out("test.vtk");
 grid_out.write_vtk(tria, out);

}


template <int dim>
void
Interpolator<dim>::setup_dofs()
{
 dof_handler.distribute_dofs(fe);
 std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
           << std::endl;


 data_vector.reinit(dof_handler.n_dofs());
}


template <int dim>
void
Interpolator<dim>::interpolate()
{
 SineFunction<dim> func;
 VectorTools::interpolate(dof_handler, func, data_vector);
}


template <int dim>
void
Interpolator<dim>::calculate_l2_error()
{
 // Analytical solution
 SineFunction<dim> func;


 // Create a Gauss Quadrature with the appropriate dimension and degree
 const QGauss<1> quadrature_formula(fe.degree + 1);


 // Here, we create an FEValues object of dimension 1 by passing the finite
 // element, the quadrature formula and other flags as parameters.
 FEValues<1> fe_values(fe,
                       quadrature_formula,
                       update_values | update_JxW_values |
                         update_quadrature_points);


 // Extract the number of quadrature points from the fe_values object
 const unsigned int n_q_points = fe_values.n_quadrature_points;


 // Create a variable to store the integration result
 double error = 0;


 // Create an std::vector which will contain all the interpolated values at the
 // gauss points
 std::vector<double> values(n_q_points);

//  int i_cell = 0;
 // Now we loop over all the cells
 for (auto cell : dof_handler.active_cell_iterators())
   {
     // This call reinitializes the valuess, determinants, and other relevant
     // information for the given cell. This is an expensive but needed call.
     fe_values.reinit(cell);
     fe_values.get_function_values(data_vector, values);
     // QUESTION PABLO: why this is inside the loop if at each iteration is the same vector ??
     
    //  std::cout<< "i_cell: "<<i_cell<<std::endl;
    //  ++i_cell;
    //  std::cout<<data_vector<<std::endl;



     // Loop over all quadrature points of each cell
     for (unsigned int q = 0; q < n_q_points; q++)
       {
         double analytical_solution = func.value(fe_values.get_quadrature_points()[q]);        

        //  //************************
        //  // TODO Calculate the L2 norm of the error
        //  //************************
        if (1==1){
            std::cout<<"Point : "<< fe_values.get_quadrature_points()[q]<<std::endl;
            std::cout<<values[q]<<std::endl;
            std::cout<<analytical_solution<<std::endl;
            std::cout<<""<<std::endl;
        }
        error += (values[q] - analytical_solution) * (values[q] - analytical_solution);
        
        //  Vector<double> tmp_vector(1);
        //  VectorTools::point_value(dof_handler, data_vector, fe_values.get_quadrature_points()[q], tmp_vector);

        //  //std::cout<< "Point: "<<fe_values.get_quadrature_points()[q]<< ",  Interpolation: "<<std::pow(tmp_vector[0] - analytical_solution,2) <<std::endl;
        //  error += std::pow(tmp_vector[0] - analytical_solution,2);

       }
   }
error = std::sqrt(error/((dof_handler.n_dofs()-1)*2));
 
// std::cout << (dof_handler.n_dofs()-1)*2 << std::endl;
std::cout << " - error(l2) : " << error << std::endl;


}


template <int dim>
void
Interpolator<dim>::output_results()
{
 std::string prefix("results_" + Utilities::int_to_string(order, 2) + "_" +
                    Utilities::int_to_string(number_of_initial_refinement, 2));
 std::string vtk_filename(prefix + ".plt");
 std::string dat_filename(prefix + ".dat");


 // Write the interpolation result in vtu format
 DataOut<dim> data_out;
 data_out.attach_dof_handler(dof_handler);
 Vector<double>     data_vector2;
 data_vector2.reinit(dof_handler.n_dofs());
 data_out.add_data_vector(data_vector, "solution");
 data_out.add_data_vector(data_vector2, "mesh");
 data_out.build_patches();
 std::ofstream output(vtk_filename);
 data_out.write_tecplot(output);


 // Calculate raw interpolation data in .dat format
 std::ofstream f(dat_filename);
 f << "x interpolation(x)" << std::endl;
 f << std::scientific;
 Point<dim> p;
 p[0]     = 0;
 double L = 10;




 for (unsigned int i = 0; i <= 1000; ++i)
   {
     p[0] = L * i / 1000.0;


     Vector<double> tmp_vector(1);
     VectorTools::point_value(dof_handler, data_vector, p, tmp_vector);
     f << p[0] << " " << tmp_vector[0];
     f << std::endl;
   }
}


template <int dim>
void
Interpolator<dim>::run()
{
 setup_triangulation();
 setup_dofs();
 interpolate();
 calculate_l2_error();
 output_results();
}


int
main()
{
 //************************
 // TODO
 // Use the Interpolator class with the specified orders and refinement levels
 //************************


 // an example:
 Interpolator<1> test(1,1);
 test.run();

 // Will instantiate the class using order 1 and 3 global refinements
 // Use for loops to automatize this running


//  for (unsigned int Qi = 1; Qi <= 3; ++Qi)
//  {
   
//    for (unsigned int ilvl = 1; ilvl <=6; ++ilvl){


//      std::cout << "Interpolation order (Qi): "<<Qi<<std::endl;


//      std::cout << "Level of refinement :"<<ilvl<<std::endl;


//      Interpolator<1> test(Qi, ilvl);
//      test.run();


//      std::cout << " " << std::endl;
//    }
//  }
 }



