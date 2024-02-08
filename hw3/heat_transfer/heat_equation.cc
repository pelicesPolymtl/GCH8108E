/* ----------------------------------------------------------------------------
 * This problem is to finally solve a PDE using the Finite Element Method with
 * deal.II. Specifically the heat equation in 1D and 2D, and an additional 
 * problem of your choice in 3D.
 * ---------------------------------------------------------------------
 */

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

// Function for the right hand side of both the 1D and 2D problems
template <int dim>
class RightHandSide : public Function<dim>
{
public:
  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;
};

// TODO: complete the value function. Tip: Use an if for the different dimensions.
template <int dim>
double
RightHandSide<dim>::value(const Point<dim> &p,
                          const unsigned int /*component*/) const
{
    if(dim == 1){
        return -1;
    }
    else return 0.0;
}

// Function for the analytical solution of both the 1D and 2D problems
template <int dim>
class AnalyticalSolution : public Function<dim>
{
public:
  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;
};

// TODO: complete the value function. Tip: Use an if for the different dimensions.
template <int dim>
double
AnalyticalSolution<dim>::value(const Point<dim> &p,
                               const unsigned int /*component*/) const
{
    if(dim == 1){
        return p[0]*p[0]/2.*(p[0]-5.);
    }
    else return 0.0;
}

// Main class to solve the heat equation in different dimensions
template <int dim>
class HeatEquation
{
public:
  HeatEquation();
  void
  run();

private:
  /**
   * @brief setup_triangulation Sets-up the triangulation. This is where you will create the grid
   * // TODO - Go and complete this function
   */
  void
  setup_triangulation();

  /**
   * @brief setup_system This part generates the sparsity pattern and allocates the memory for the matrix, the right-hand side and the solution.
   */
  void
  setup_system();

  /**
   * @brief assemble_system Assembles the matrix and the right-hand side
   * // TODO - Go and complete this function
   */
  void
  assemble_system();

  /**
   * @brief Solve_linear_system Solves the linear system of equation that arise from the problem
   */
  void
  solve_linear_system();

  /**
   * @brief Solve_linear_system Solves the linear system of equation that arise from the problem
   * // TODO - Go and complete this function
   */
  void
  calculate_L2_error();

  /**
   * @brief output_results Outputs the results of the simulation into vtk files for Paraview.
   * // TODO - Go and complete this function
   */
  void
  output_results() const;

  Triangulation<dim> triangulation;
  FE_Q<dim>          fe;
  DoFHandler<dim>    dof_handler;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double> solution;
  Vector<double> system_rhs;
};


template <int dim>
HeatEquation<dim>::HeatEquation()
  : fe(1)
  , dof_handler(triangulation)
{}

template <int dim>
void
HeatEquation<dim>::setup_triangulation()
{
  // We use the dimensions to seperate the three cases
  // This is not very clean code. The best idea would have been to have a case
  // parameter. But for the purpose of this homework, this will be sufficient.
  if (dim == 1)
    {
      // TODO: Generate the mesh for the 1D problem and refine it globally. Tip: Use the hyper cube function.
      int number_of_initial_refinement = 3;
      GridGenerator::hyper_cube(triangulation, 0., 5.);
      triangulation.refine_global(number_of_initial_refinement);
    }

  if (dim == 2)
    {
      // TODO: Generate the mesh for the 1D problem and refine it globally. Tip: Use the hyper shell function.
      int number_of_initial_refinement = 3;
      GridGenerator::hyper_cube(triangulation, 0., 5.);
      triangulation.refine_global(number_of_initial_refinement);

    }
  if (dim == 3)
    {}

  std::cout << "   Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "   Total number of cells: " << triangulation.n_cells()
            << std::endl;
}

template <int dim>
void
HeatEquation<dim>::setup_system()
{
  // This call enumerates the DoFs
  dof_handler.distribute_dofs(fe);

  std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;

  // Here, we create a sparsity pattern, i.e. a structure used to store
  // the places of non-zero elements. Then it is passed to the system_matrix.
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);

  // Here we set the appropriate sizes of the solution vector and the right hand side
  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}

template <int dim>
void
HeatEquation<dim>::calculate_L2_error()
{
  // TODO: complete this function similarly to the analogous
  // function in the interpolation exercise.

    for (auto &cell: triangulation.active_cell_iterators())
    {
      // Check criteria using cell->center(). Tip: the center of a cell in 2D 
      // is a point. 
      for (unsigned int j=0, j < fe->n_quadrature_points; j++ )
      {
        std::cout<<cell->quad[j]<<std::endl;
      }

    }

}

template <int dim>
void
HeatEquation<dim>::assemble_system()
{
  QGauss<dim> quadrature_formula(fe.degree + 1);

  // We want to have a non-constant right hand side, so we use an object of
  // the class declared above to generate the necessary data. Since this right
  // hand side object is only used locally in the present function, we declare
  // it here as a local variable:
  RightHandSide<dim> right_hand_side;

  // To solve our problem, we will need to have the values of the shape
  // function, the gradient of the shape function, the location of the
  // quadrature points and the JxW values. Thus we update these fields with the
  // FeValues class.
  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  // We need to know how many degrees of freedom we have per cell.
  // The degrees of freedom are the colocation point, the points which
  // we use to carry out our interpolation.
  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();


  // We create cell matrices and cell rhs to store the information
  // of the integral over the cell as we are building it
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  // We will store the index in the "big" matrix of the degrees of freedom
  // in the cell using this vector here. Don't be afraid of
  // types::global_dof_index It is nothing more than an unsigned integer (an
  // integer without a sign bit)
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // Next, we again have to loop over all cells and assemble local
  // contributions.  Note, that a cell is a quadrilateral in two space
  // dimensions, but a hexahedron in 3D. In fact, the
  // <code>active_cell_iterator</code> data type is something different,
  // depending on the dimension we are in, but to the outside world they look
  // alike and you will probably never see a difference.
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell_matrix = 0;
      cell_rhs    = 0;

      // Now we have to assemble the local matrix and right hand side.
      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        for (const unsigned int i : fe_values.dof_indices())
          {
            for (const unsigned int j : fe_values.dof_indices())
              // TODO: assemble the cell_matrix for the component i j. Tip: use the fe_values.shape_grad(...)
              cell_matrix(i, j) +=
                    (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                     fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                     fe_values.JxW(q_index)); 

            // Here we assemble the right hand side using the
            // right_hand_side function which uses the position of the
            // quadrature point This means that we could have a RHS that depends
            // on x,y,z if we wished it to. This would automatically be taken
            // into account.
            const auto &x_q = fe_values.quadrature_point(q_index);

            // TODO: assemble the rhs for the component it. Tip: use fe_values.shape_value(...)
            cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                            right_hand_side.value(x_q) * // f(x_q)
                            fe_values.JxW(q_index)); // dx
          }

      // Here we simply add each cell matrix to the global system matrix
      cell->get_dof_indices(local_dof_indices);
      for (const unsigned int i : fe_values.dof_indices())
        {
          for (const unsigned int j : fe_values.dof_indices())
            system_matrix.add(local_dof_indices[i],
                              local_dof_indices[j],
                              cell_matrix(i, j));

          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }

  // As the final step in this function, we wanted to have non-homogeneous
  // boundary values and homogenous boundary values. These boundary
  // values are imposed using the VectorTool utilities.
  //
  // The function VectorTools::interpolate_boundary_values() will only work
  // on faces that have been marked with boundary indicator 0 and 1.  For
  // the Laplace equation doing nothing is equivalent to assuming that
  // on those parts of the boundary a zero Neumann boundary condition holds. So
  // if we had boundary conditions with number 2 or 3, right now they would
  // have Neumann boundary conditions imposed on them.
  // Tip: to understand better how boundary ids are set for each triangulation
  // check the documentation for the hyper_cube and the hyper_shell. 

  std::map<types::global_dof_index, double> boundary_values;
  if (dim == 1)
    {
      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               Functions::ZeroFunction<dim>(),
                                               boundary_values);
      VectorTools::interpolate_boundary_values(dof_handler,
                                               1,
                                               Functions::ZeroFunction<dim>(),
                                               boundary_values);
    }
  if (dim == 2)
    {
      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               Functions::ZeroFunction<dim>(),
                                               boundary_values);
      VectorTools::interpolate_boundary_values(
        dof_handler, 1, Functions::ConstantFunction<dim>(1.), boundary_values);
    }

  // This applies the boundary conditions on the system matrix.
  MatrixTools::apply_boundary_values(boundary_values,
                                     system_matrix,
                                     solution,
                                     system_rhs);
}

template <int dim>
void
HeatEquation<dim>::solve_linear_system()
{
  // We use a sparse direct solver to solve the equations
  // We could also use an iterative solver. It would have been more efficient.

  SparseDirectUMFPACK A_direct;
  A_direct.initialize(system_matrix);
  A_direct.vmult(solution, system_rhs);

  // Try it out if you want by uncommenting what follows! :)

  SolverControl solver_control(1000, 1e-12);   
  SolverCG<Vector<double>> solver(solver_control);
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());

  // We have made one addition, though: since we suppress output from the
  // linear solvers, we have to print the number of iterations by hand.
  std::cout << "   " << solver_control.last_step()
            << " CG iterations needed to obtain convergence." << std::endl;
    
}

template <int dim>
void
HeatEquation<dim>::output_results() const
{
  // TODO: you should now be familiar with how outputting data works.
  // Create the DataOut object and write the solution into a vtu file.
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches();
  std::ofstream output("test.plt");
  data_out.write_tecplot(output);
}

template <int dim>
void
HeatEquation<dim>::run()
{
  std::cout << "Solving problem in " << dim << " space dimensions."
            << std::endl;

  setup_triangulation();
  setup_system();
  assemble_system();
  solve_linear_system();
  calculate_L2_error();
  output_results();
}

int
main()
{
  // Here we create a heat equation object in 1D and 2D
  // and solve it by using the function run(). Go to the heat
  // equation class and start completing the code.
  {
    HeatEquation<1> laplace_problem_1d;
    laplace_problem_1d.run();
  }
//   {
//     HeatEquation<2> laplace_problem_2d;
//     laplace_problem_2d.run();
//   }

  {
    // TODO: solve a problem of your choice here

  }

  return 0;
}
