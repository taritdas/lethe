#include <core/manifolds.h>
#include <core/pvd_handler.h>
#include <core/simulation_control.h>
#include <core/solutions_output.h>

#include <solvers/fluid_dynamics_matrix_based.h>
#include <solvers/solid_phase.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/vector_tools.h>

#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>


// using namespace dealii;



class SolidBlockDiagonalPreconditioner : public Subscriptor
{
public:
  SolidBlockDiagonalPreconditioner(
    const std::shared_ptr<TrilinosWrappers::PreconditionAMG> &velocity_amg,
    const std::shared_ptr<TrilinosWrappers::PreconditionILU> &velocity_ilu,
    const std::shared_ptr<TrilinosWrappers::PreconditionILU> &alpha_ilu,
    const bool                                                use_amg)
    : velocity_amg(velocity_amg)
    , velocity_ilu(velocity_ilu)
    , alpha_ilu(alpha_ilu)
    , use_amg(use_amg)
  {}

  void
  vmult(TrilinosWrappers::MPI::BlockVector       &dst,
        const TrilinosWrappers::MPI::BlockVector &src) const
  {
    if (dst.block(0).size() > 0)
      {
        if (use_amg && velocity_amg)
          velocity_amg->vmult(dst.block(0), src.block(0));
        else if (velocity_ilu)
          velocity_ilu->vmult(dst.block(0), src.block(0));
        else
          dst.block(0) = src.block(0);
      }

    if (dst.block(1).size() > 0)
      {
        if (alpha_ilu)
          alpha_ilu->vmult(dst.block(1), src.block(1));
        else
          dst.block(1) = src.block(1);
      }
  }

private:
  std::shared_ptr<TrilinosWrappers::PreconditionAMG> velocity_amg;
  std::shared_ptr<TrilinosWrappers::PreconditionILU> velocity_ilu;
  std::shared_ptr<TrilinosWrappers::PreconditionILU> alpha_ilu;
  bool                                               use_amg;
};

template <int dim>
void
SolidPhaseSolver<dim>::setup_linear_preconditioners()
{
  velocity_amg.reset();
  velocity_ilu.reset();
  alpha_ilu.reset();

  const bool use_amg = (parameters.amg_sweeps > 0);

  if (use_amg)
    {
      velocity_amg = std::make_shared<TrilinosWrappers::PreconditionAMG>();

      TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
      amg_data.elliptic              = parameters.amg_elliptic;
      amg_data.higher_order_elements = (order > 1);
      amg_data.smoother_type         = parameters.amg_smoother_type.c_str();
      amg_data.smoother_sweeps       = parameters.amg_sweeps;
      amg_data.aggregation_threshold = parameters.amg_agg_threshold;
      amg_data.output_details        = false;

      velocity_amg->initialize(system_matrix.block(0, 0), amg_data);
    }
  else
    {
      velocity_ilu = std::make_shared<TrilinosWrappers::PreconditionILU>();

      TrilinosWrappers::PreconditionILU::AdditionalData ilu_data;
      ilu_data.overlap = parameters.ilu_overlap;

      velocity_ilu->initialize(system_matrix.block(0, 0), ilu_data);
    }

  alpha_ilu = std::make_shared<TrilinosWrappers::PreconditionILU>();

  TrilinosWrappers::PreconditionILU::AdditionalData ilu_data;
  ilu_data.overlap = parameters.ilu_overlap;
  alpha_ilu->initialize(system_matrix.block(1, 1), ilu_data);
}



template <int dim>
class SolidVelocityFunctionDefined : public Function<dim>
{
public:
  SolidVelocityFunctionDefined(Function<dim> *u,
                               Function<dim> *v,
                               Function<dim> *w = nullptr)
    : Function<dim>(dim + 1)
    , u(u)
    , v(v)
    , w(w)
  {}

  double
  value(const Point<dim> &p, const unsigned int component) const override
  {
    if (component == 0)
      return u->value(p);
    if (component == 1 && dim > 1)
      return v->value(p);
    if (component == 2 && dim > 2)
      return w->value(p);
    return 0.0;
  }

private:
  Function<dim> *u;
  Function<dim> *v;
  Function<dim> *w;
};

template <int dim>
class SolidAlphaFunctionDefined : public Function<dim>
{
public:
  explicit SolidAlphaFunctionDefined(Function<dim> *alpha_function)
    : Function<dim>(dim + 1)
    , alpha_function(alpha_function)
  {}

  double
  value(const Point<dim> &p, const unsigned int component) const override
  {
    if (component == dim)
      return alpha_function->value(p);
    return 0.0;
  }

private:
  Function<dim> *alpha_function;
};


struct SolidKTGFValues
{
  double radial_distribution;
  double particle_pressure;
  double particle_pressure_derivative;
  double shear_viscosity;
  double bulk_viscosity;
  double conductivity;
  double collisional_dissipation_coefficient;
  double frictional_pressure;
  double frictional_pressure_derivative;
};


template <int dim>
SolidKTGFValues
evaluate_solid_ktgf(const SolidPhaseParameters<dim> &parameters,
                    const double                     alpha,
                    const double                     theta,
                    const double                     velocity_divergence)
{
  const double rho_s = parameters.rho_s;
  const double d_p   = parameters.particle_diameter;
  const double e_p   = parameters.restitution_coefficient;
  const double a_max = parameters.alpha_max;


  AssertThrow(std::isfinite(alpha),
              ExcMessage("KTGF received non-finite alpha."));

  AssertThrow(std::isfinite(theta),
              ExcMessage("KTGF received non-finite theta."));

  AssertThrow(theta >= 0.0,
              ExcMessage(
                "KTGF received negative granular temperature: theta = " +
                std::to_string(theta)));



  AssertThrow(parameters.particle_diameter > 0.0,
              ExcMessage("Particle diameter must be positive."));

  // AssertThrow(alpha >= 0.0,
  //             ExcMessage(
  //               "KTGF received negative solid volume fraction: alpha = " +
  //               std::to_string(alpha)));


  AssertThrow(alpha >= -1e-14,
              ExcMessage(
                "KTGF received negative solid volume fraction: alpha = " +
                std::to_string(alpha)));

  AssertThrow(parameters.alpha_max > 0.0,
              ExcMessage("alpha_max must be positive."));

  AssertThrow(parameters.rho_s > 0.0,
              ExcMessage("Solid density must be positive."));

  AssertThrow(parameters.restitution_coefficient >= 0.0 &&
                parameters.restitution_coefficient <= 1.0,
              ExcMessage(
                "Restitution coefficient must be between zero and one."));

  AssertThrow(std::isfinite(velocity_divergence),
              ExcMessage(
                "KTGF received non-finite solid velocity divergence."));


  /*
   * Dense-phase transition.
   *
   * KTGF radial distribution is bounded at alpha_friction.
   * Johnson-Jackson frictional pressure becomes active above
   * alpha_friction.
   */
  // const double alpha_friction = 0.58;

  // AssertThrow(alpha_friction > 0.0 && alpha_friction < a_max,
  //             ExcMessage(
  //               "alpha_friction must satisfy 0 < alpha_friction <
  //               alpha_max."));


  // /*
  //  * Sinclair-Jackson radial distribution function.
  //  *
  //  * g0 uses the actual alpha in the dilute regime and is
  //  * capped at alpha_friction in the dense/frictional regime.
  //  */
  // const double alpha_g0 = std::min(alpha, alpha_friction);

  // const double r = std::cbrt(alpha_g0 / a_max);

  // const double one_minus_r = 1.0 - r;

  // AssertThrow(one_minus_r > 0.0,
  //             ExcMessage("Invalid radial-distribution denominator."));

  // const double g0 = 1.0 / one_minus_r;


  // /*
  //  * Regularized derivative of g0.
  //  *
  //  * The small lower bound avoids the singular expression
  //  * in g0' as alpha -> 0.
  //  *
  //  * This follows the Sinclair-Jackson implementation strategy.
  //  */
  // const double alpha_g0_prime = std::min(std::max(alpha, 1e-3),
  // alpha_friction);

  // const double r_prime = std::cbrt(alpha_g0_prime / a_max);

  // const double one_minus_r_prime = 1.0 - r_prime;

  // const double dg0_dalpha = 1.0 / (3.0 * a_max * r_prime * r_prime *
  //                                  one_minus_r_prime * one_minus_r_prime);

  // // [REVIEW FIX 1] g0 is evaluated at min(alpha, alpha_friction), so it is
  // // constant above alpha_friction and its derivative there is zero.
  // // Before: G_kin was ~7x too large for alpha > 0.58 and the momentum force
  // // G_kin grad(alpha) + C_p grad(Theta) was not grad(p_kin).
  // const double alpha_squared_dg0_dalpha =
  //   (alpha < alpha_friction) ? alpha * alpha * dg0_dalpha : 0.0;

  // /*
  //  * Johnson-Jackson frictional pressure.
  //  */
  // const double Fr              = 0.05; // Pa
  // const double eta_friction    = 2.0;
  // const double p_friction_exp  = 5.0;
  // const double alpha_delta_min = 0.01;

  // AssertThrow(alpha_delta_min > 0.0,
  //             ExcMessage("alpha_delta_min must be positive."));


  // const double alpha_c = a_max - alpha_delta_min;

  // AssertThrow(alpha_c > alpha_friction,
  //             ExcMessage("alpha_max - alpha_delta_min must exceed "
  //                        "alpha_friction."));

  // const auto jj_pressure = [&](const double a) {
  //   const double da = std::max(a - alpha_friction, 0.0);
  //   return Fr * std::pow(da, eta_friction) /
  //          std::pow(a_max - a, p_friction_exp);
  // };

  // const auto jj_derivative = [&](const double a) {
  //   if (a <= alpha_friction)
  //     return 0.0;
  //   const double da  = a - alpha_friction;
  //   const double gap = a_max - a;
  //   return Fr *
  //          (eta_friction * std::pow(da, eta_friction - 1.0) * gap +
  //           p_friction_exp * std::pow(da, eta_friction)) /
  //          std::pow(gap, p_friction_exp + 1.0);
  // };

  // double frictional_pressure            = 0.0;
  // double frictional_pressure_derivative = 0.0;

  // if (alpha <= alpha_c)
  //   {
  //     frictional_pressure            = jj_pressure(alpha);
  //     frictional_pressure_derivative = jj_derivative(alpha);
  //   }
  // else
  //   {
  //     const double G_c = jj_derivative(alpha_c);

  //     frictional_pressure = jj_pressure(alpha_c) + G_c * (alpha - alpha_c);
  //     frictional_pressure_derivative = G_c;
  //   }


  /*
   * Radial distribution function: Carnahan-Starling, P&F Eq. (14).
   * Finite for every alpha < 1, so no cap and no regularisation needed.
   */
  const double one_minus_a = 1.0 - alpha;

  AssertThrow(one_minus_a > 0.0,
              ExcMessage("Carnahan-Starling g0 needs alpha < 1."));

  const double om2 = one_minus_a * one_minus_a;
  const double om3 = om2 * one_minus_a;
  const double om4 = om3 * one_minus_a;

  const double g0 =
    1.0 / one_minus_a + 1.5 * alpha / om2 + 0.5 * alpha * alpha / om3;

  // d g0 / d alpha (exact derivative of Eq. 14)
  const double dg0_dalpha =
    2.5 / om2 + 4.0 * alpha / om3 + 1.5 * alpha * alpha / om4;

  const double alpha_squared_dg0_dalpha = alpha * alpha * dg0_dalpha;

  /*
   * Frictional normal stress, P&F Eq. (15):
   *   p_f = 1e25 (alpha - alpha_fr_min)^10   for alpha > alpha_fr_min,
   *   dp_f/dalpha = 1e26 (alpha - alpha_fr_min)^9.
   * alpha_fr_min = 0.6 is the packing value of P&F's settling case
   * ("maximum allowed value, fixed at 0.6").
   * No frictional viscosity: P&F only give the normal stress.
   */
  const double alpha_fr_min = 0.6;

  const double da_fr = std::max(alpha - alpha_fr_min, 0.0);

  const double frictional_pressure            = 1e25 * std::pow(da_fr, 10);
  const double frictional_pressure_derivative = 1e26 * std::pow(da_fr, 9);



  const double sqrt_theta_over_pi = std::sqrt(theta / numbers::PI);

  const double sqrt_theta_pi = std::sqrt(theta * numbers::PI);

  SolidKTGFValues values;

  values.radial_distribution = g0;

  values.frictional_pressure = frictional_pressure;

  values.frictional_pressure_derivative = frictional_pressure_derivative;


  /*
   * KTGF particle pressure.
   *
   * Keep frictional pressure OUT of this quantity because
   * particle_pressure is also used in the granular-temperature
   * production term.
   */
  const double kinetic_particle_pressure =
    rho_s * alpha * theta +
    2.0 * rho_s * alpha * alpha * g0 * theta * (1.0 + e_p);

  values.particle_pressure = kinetic_particle_pressure;


  /*
   * KTGF pressure derivative.
   */
  const double kinetic_pressure_derivative =
    rho_s * theta *
    (1.0 + 4.0 * alpha * g0 * (1.0 + e_p) +
     2.0 * (1.0 + e_p) * alpha_squared_dg0_dalpha);


  /*
   * Momentum particle-pressure derivative:
   *
   * dp_total/dalpha
   * =
   * dp_KTGF/dalpha
   * +
   * dp_friction/dalpha.
   */
  values.particle_pressure_derivative = kinetic_pressure_derivative;



  const double mu_collisional = (4.0 / 5.0) * alpha * alpha * rho_s * d_p * g0 *
                                (1.0 + e_p) * sqrt_theta_over_pi;

  const double kinetic_factor = 1.0 + (4.0 / 5.0) * (1.0 + e_p) * alpha * g0;

  const double mu_kinetic = 10.0 * rho_s * d_p * sqrt_theta_pi /
                            (96.0 * g0 * (1.0 + e_p)) * kinetic_factor *
                            kinetic_factor;

  values.shear_viscosity = mu_collisional + mu_kinetic;

  values.bulk_viscosity = (4.0 / 3.0) * alpha * alpha * rho_s * d_p * g0 *
                          (1.0 + e_p) * sqrt_theta_over_pi;

  const double conductivity_factor =
    1.0 + (6.0 / 5.0) * g0 * alpha * (1.0 + e_p);

  values.conductivity =
    150.0 * rho_s * d_p * sqrt_theta_pi / (384.0 * g0 * (1.0 + e_p)) *
      (conductivity_factor * conductivity_factor) +
    2.0 * alpha * alpha * rho_s * d_p * g0 * (1.0 + e_p) * sqrt_theta_over_pi;

  // values.collisional_dissipation_coefficient =
  //   3.0 * (1.0 - e_p * e_p) * rho_s * alpha * alpha * g0 *
  //   ((4.0 / d_p) * sqrt_theta_over_pi - velocity_divergence);

  values.collisional_dissipation_coefficient =
    3.0 * (1.0 - e_p * e_p) * rho_s * alpha * alpha * g0 *
    ((4.0 / d_p) * sqrt_theta_over_pi);

  return values;
}


template <int dim>
Tensor<2, dim>
make_solid_stress(const double          mu_s,
                  const double          lambda_s,
                  const Tensor<2, dim> &velocity_gradient)
{
  Tensor<2, dim> identity;
  for (unsigned int d = 0; d < dim; ++d)
    identity[d][d] = 1.0;

  const double div_u = trace(velocity_gradient);

  return mu_s * (velocity_gradient + transpose(velocity_gradient)) +
         (lambda_s - (2.0 / 3.0) * mu_s) * div_u * identity;
}


template <int dim>
SolidPhaseSolver<dim>::SolidPhaseSolver(
  const SolidPhaseParameters<dim>           &p,
  const std::shared_ptr<SimulationControl>  &simulation_control_in,
  parallel::distributed::Triangulation<dim> &tria,
  MPI_Comm                                   comm)
  : parameters(p)
  , mpi_communicator(comm)
  , simulation_control(simulation_control_in)
  , order(parameters.order)
  , fe(FE_Q<dim>(order), dim, FE_Q<dim>(order), 1)
  , theta_fe(order)
  , triangulation(tria)
  , dof_handler(triangulation)
  , theta_dof_handler(triangulation)
  , rho_s(parameters.rho_s)
  , beta(parameters.beta)
  , timestep_number(0)
  , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
  , computing_timer(mpi_communicator,
                    pcout,
                    TimerOutput::never,
                    TimerOutput::wall_times)
  , output_every(parameters.output_every)
  , digits(parameters.digits)
  , output_folder(parameters.output_folder)
  , output_prefix(parameters.output_prefix)
{
  AssertThrow(simulation_control != nullptr,
              ExcMessage(
                "simulation_control is null in SolidPhaseSolver constructor."));
}



template <int dim>
void
SolidPhaseSolver<dim>::setup_dofs()
{
  if (parameters.verbose_assembly)
    pcout << "setting up dofs...\n" << std::endl;

  TimerOutput::Scope t(computing_timer, "setup");

  dof_handler.distribute_dofs(fe);

  std::vector<unsigned int> block_component(fe.n_components(), 0);
  block_component[dim] = 1;
  DoFRenumbering::component_wise(dof_handler, block_component);

  const auto dofs_per_block =
    DoFTools::count_dofs_per_fe_block(dof_handler, block_component);

  AssertDimension(dofs_per_block.size(), 2);

  const types::global_dof_index n_u = dofs_per_block[0];
  const types::global_dof_index n_a = dofs_per_block[1];

  pcout << "   Number of orders of freedom: " << dof_handler.n_dofs() << " ("
        << n_u << "+" << n_a << ")" << std::endl;

  locally_owned    = dof_handler.locally_owned_dofs();
  locally_relevant = DoFTools::extract_locally_relevant_dofs(dof_handler);

  owned_partitioning.clear();
  relevant_partitioning.clear();

  owned_partitioning.emplace_back(locally_owned.get_view(0, n_u));
  owned_partitioning.emplace_back(locally_owned.get_view(n_u, n_u + n_a));

  relevant_partitioning.emplace_back(locally_relevant.get_view(0, n_u));
  relevant_partitioning.emplace_back(locally_relevant.get_view(n_u, n_u + n_a));

  constraints.clear();
  constraints.reinit(locally_owned, locally_relevant);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);

  AssertThrow(simulation_control != nullptr,
              ExcMessage("simulation_control is null in setup_dofs()."));

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar alpha(dim);

  const ComponentMask vel_mask   = fe.component_mask(velocities);
  const ComponentMask alpha_mask = fe.component_mask(alpha);

  const double time = simulation_control->get_current_time();

  for (const auto &[id, type] : parameters.boundary_conditions.type)
    {
      if (type == BoundaryConditions::BoundaryType::noslip)
        {
          VectorTools::interpolate_boundary_values(dof_handler,
                                                   id,
                                                   Functions::ZeroFunction<dim>(
                                                     dim + 1),
                                                   constraints,
                                                   vel_mask);
        }
      else if (type == BoundaryConditions::BoundaryType::slip)
        {
          std::set<types::boundary_id> slip_boundaries;
          slip_boundaries.insert(id);

          VectorTools::compute_no_normal_flux_constraints(dof_handler,
                                                          0,
                                                          slip_boundaries,
                                                          constraints);
        }
      else if (type == BoundaryConditions::BoundaryType::function)
        {
          auto &velocity_functions =
            *parameters.boundary_conditions.velocity_functions.at(id);

          velocity_functions.u.set_time(time);
          velocity_functions.v.set_time(time);
          if constexpr (dim == 3)
            velocity_functions.w.set_time(time);

          auto &alpha_function =
            *parameters.boundary_conditions.alpha_functions.at(id);
          alpha_function.set_time(time);

          SolidVelocityFunctionDefined<dim> velocity_bc(
            &velocity_functions.u,
            &velocity_functions.v,
            (dim == 3 ? &velocity_functions.w : nullptr));

          SolidAlphaFunctionDefined<dim> alpha_bc(&alpha_function);

          VectorTools::interpolate_boundary_values(
            dof_handler, id, velocity_bc, constraints, vel_mask);

          VectorTools::interpolate_boundary_values(
            dof_handler, id, alpha_bc, constraints, alpha_mask);
        }
      else if (type == BoundaryConditions::BoundaryType::outlet)
        {
          // Natural boundary condition.

          // Outlet flux terms are assembled in assemble_boundary_face().
        }


      else if (type == BoundaryConditions::BoundaryType::periodic)
        {
          DoFTools::make_periodicity_constraints(
            dof_handler,
            id,
            parameters.boundary_conditions.periodic_neighbor_id.at(id),
            parameters.boundary_conditions.periodic_direction.at(id),
            constraints);
        }
      else
        {
          AssertThrow(
            false,
            ExcMessage(
              "Unsupported solid boundary type in setup_dofs() for boundary ID " +
              std::to_string(static_cast<unsigned int>(id))));
        }
    }

  constraints.close();

  system_matrix.clear();

  TrilinosWrappers::BlockSparsityPattern sp(owned_partitioning,
                                            owned_partitioning,
                                            relevant_partitioning,
                                            mpi_communicator);

  DoFTools::make_sparsity_pattern(dof_handler,
                                  sp,
                                  constraints,
                                  false,
                                  Utilities::MPI::this_mpi_process(
                                    mpi_communicator));

  sp.compress();
  system_matrix.reinit(sp);

  alpha_low_matrix.reinit(sp);

  alpha_consistent_mass_matrix.reinit(sp);
  alpha_diffusion_matrix.reinit(sp);

  solution.reinit(owned_partitioning, mpi_communicator);
  old_solution.reinit(owned_partitioning, mpi_communicator);
  older_solution.reinit(owned_partitioning, mpi_communicator);
  system_rhs.reinit(owned_partitioning, mpi_communicator);

  alpha_low_rhs.reinit(owned_partitioning, mpi_communicator);

  alpha_low_solution.reinit(owned_partitioning, mpi_communicator);


  alpha_fct_solution.reinit(owned_partitioning[1], mpi_communicator);

  fct_R_plus.reinit(owned_partitioning[1], mpi_communicator);

  fct_R_minus.reinit(owned_partitioning[1], mpi_communicator);

  alpha_low_relevant.reinit(owned_partitioning[1],
                            relevant_partitioning[1],
                            mpi_communicator);

  fct_R_plus_relevant.reinit(owned_partitioning[1],
                             relevant_partitioning[1],
                             mpi_communicator);

  fct_R_minus_relevant.reinit(owned_partitioning[1],
                              relevant_partitioning[1],
                              mpi_communicator);

  fct_P_plus.reinit(owned_partitioning[1], mpi_communicator);

  fct_P_minus.reinit(owned_partitioning[1], mpi_communicator);

  fct_correction.reinit(owned_partitioning[1], mpi_communicator);

  fct_lumped_mass.reinit(owned_partitioning[1], mpi_communicator);

  picard_solution.reinit(owned_partitioning, mpi_communicator);

  locally_relevant_solution.reinit(owned_partitioning,
                                   relevant_partitioning,
                                   mpi_communicator);

  locally_relevant_old_solution.reinit(owned_partitioning,
                                       relevant_partitioning,
                                       mpi_communicator);

  locally_relevant_older_solution.reinit(owned_partitioning,
                                         relevant_partitioning,
                                         mpi_communicator);

  locally_relevant_picard_solution.reinit(owned_partitioning,
                                          relevant_partitioning,
                                          mpi_communicator);

  locally_relevant_momentum_solution.reinit(owned_partitioning,
                                            relevant_partitioning,
                                            mpi_communicator);

  // [REVIEW FIX 5] lagged FCT correction delta = alpha^F - alpha^L
  // (block 1; block 0 stays zero). Declare it in solid_phase.h next to
  // locally_relevant_momentum_solution.
  locally_relevant_fct_delta.reinit(owned_partitioning,
                                    relevant_partitioning,
                                    mpi_communicator);
  locally_relevant_fct_delta = 0.0;

  picard_solution_ptr = nullptr;
}

template <int dim>
void
SolidPhaseSolver<dim>::setup_granular_temperature_dofs()
{
  if (parameters.verbose_assembly)
    pcout << "setting up granular-temperature dofs...\n" << std::endl;

  TimerOutput::Scope t(computing_timer, "theta setup");

  theta_dof_handler.distribute_dofs(theta_fe);

  theta_locally_owned = theta_dof_handler.locally_owned_dofs();
  theta_locally_relevant =
    DoFTools::extract_locally_relevant_dofs(theta_dof_handler);

  pcout << "   Granular-temperature degrees of freedom: "
        << theta_dof_handler.n_dofs() << std::endl;

  theta_constraints.clear();
  theta_constraints.reinit(theta_locally_owned, theta_locally_relevant);
  DoFTools::make_hanging_node_constraints(theta_dof_handler, theta_constraints);

  const double time = simulation_control->get_current_time();

  for (const auto &[id, type] : parameters.boundary_conditions.type)
    {
      if (type == BoundaryConditions::BoundaryType::function)
        {
          auto &theta_function =
            *parameters.boundary_conditions.theta_functions.at(id);
          theta_function.set_time(time);

          VectorTools::interpolate_boundary_values(theta_dof_handler,
                                                   id,
                                                   theta_function,
                                                   theta_constraints);
        }
      else if (type == BoundaryConditions::BoundaryType::periodic)
        {
          DoFTools::make_periodicity_constraints(
            theta_dof_handler,
            id,
            parameters.boundary_conditions.periodic_neighbor_id.at(id),
            parameters.boundary_conditions.periodic_direction.at(id),
            theta_constraints);
        }
      else if (type == BoundaryConditions::BoundaryType::noslip ||
               type == BoundaryConditions::BoundaryType::slip ||
               type == BoundaryConditions::BoundaryType::outlet)
        {
          // Homogeneous natural granular-temperature flux.
        }
      else
        {
          AssertThrow(false,
                      ExcMessage(
                        "Unsupported granular-temperature boundary "
                        "condition for boundary ID " +
                        std::to_string(static_cast<unsigned int>(id))));
        }
    }

  theta_constraints.close();

  TrilinosWrappers::SparsityPattern theta_sparsity(theta_locally_owned,
                                                   theta_locally_owned,
                                                   theta_locally_relevant,
                                                   mpi_communicator);

  DoFTools::make_sparsity_pattern(theta_dof_handler,
                                  theta_sparsity,
                                  theta_constraints,
                                  false,
                                  Utilities::MPI::this_mpi_process(
                                    mpi_communicator));

  theta_sparsity.compress();
  theta_matrix.reinit(theta_sparsity);

  theta_solution.reinit(theta_locally_owned, mpi_communicator);
  theta_old_solution.reinit(theta_locally_owned, mpi_communicator);
  theta_picard_solution.reinit(theta_locally_owned, mpi_communicator);
  theta_rhs.reinit(theta_locally_owned, mpi_communicator);

  locally_relevant_theta_solution.reinit(theta_locally_owned,
                                         theta_locally_relevant,
                                         mpi_communicator);
  locally_relevant_theta_old_solution.reinit(theta_locally_owned,
                                             theta_locally_relevant,
                                             mpi_communicator);
  locally_relevant_theta_picard_solution.reinit(theta_locally_owned,
                                                theta_locally_relevant,
                                                mpi_communicator);
}


template <int dim>
void
SolidPhaseSolver<dim>::set_fluid_solution_field(
  const DoFHandler<dim>               &fluid_dh,
  const Mapping<dim>                  &fluid_mapping,
  const TrilinosWrappers::MPI::Vector &fluid_solution)
{
  fluid_dof_handler_ptr    = &fluid_dh;
  fluid_mapping_ptr        = &fluid_mapping;
  fluid_solution_ptr       = &fluid_solution;
  has_fluid_velocity_field = true;
}



template <int dim>
struct SolidCopyData
{
  FullMatrix<double>                   local_matrix;
  Vector<double>                       local_rhs;
  FullMatrix<double>                   local_alpha_low_matrix;
  FullMatrix<double>                   local_alpha_convection;
  Vector<double>                       local_alpha_low_rhs;
  FullMatrix<double>                   local_alpha_consistent_mass;
  FullMatrix<double>                   local_alpha_diffusion;
  std::vector<types::global_dof_index> local_dof_indices;
  bool                                 cell_is_local = false;

  SolidCopyData(const unsigned int dofs_per_cell = 0)
    : local_matrix(dofs_per_cell, dofs_per_cell)
    , local_rhs(dofs_per_cell)
    , local_alpha_low_matrix(dofs_per_cell, dofs_per_cell)
    , local_alpha_convection(dofs_per_cell, dofs_per_cell)
    , local_alpha_low_rhs(dofs_per_cell)
    , local_alpha_consistent_mass(dofs_per_cell, dofs_per_cell)
    , local_alpha_diffusion(dofs_per_cell, dofs_per_cell)
    , local_dof_indices(dofs_per_cell)
  {}

  void
  reset()
  {
    local_matrix                = 0.0;
    local_rhs                   = 0.0;
    local_alpha_low_matrix      = 0.0;
    local_alpha_convection      = 0.0;
    local_alpha_low_rhs         = 0.0;
    local_alpha_consistent_mass = 0.0;
    local_alpha_diffusion       = 0.0;
  }
};


template <int dim>
class SolidScratchData
{
public:
  SolidScratchData(const FiniteElement<dim>  &solid_fe_in,
                   const FiniteElement<dim>  &theta_fe_in,
                   const Quadrature<dim>     &cell_quadrature_in,
                   const Quadrature<dim - 1> &face_quadrature_in,
                   const Mapping<dim>        &fluid_mapping_in,
                   const FiniteElement<dim>  &fluid_fe_in)
    : solid_fe(&solid_fe_in)
    , theta_fe(&theta_fe_in)
    , cell_quadrature(&cell_quadrature_in)
    , face_quadrature(&face_quadrature_in)
    , fluid_mapping(&fluid_mapping_in)
    , fluid_fe(&fluid_fe_in)
    , fe_values(fluid_mapping_in,
                solid_fe_in,
                cell_quadrature_in,
                update_values | update_gradients | update_hessians |
                  update_quadrature_points | update_JxW_values)
    , fe_face_values(fluid_mapping_in,
                     solid_fe_in,
                     face_quadrature_in,
                     update_values | update_normal_vectors |
                       update_quadrature_points | update_JxW_values)
    , theta_fe_values(fluid_mapping_in,
                      theta_fe_in,
                      cell_quadrature_in,
                      update_values | update_gradients)
    , fluid_fe_values(fluid_mapping_in,
                      fluid_fe_in,
                      cell_quadrature_in,
                      update_values | update_gradients)
    , n_q_points(cell_quadrature_in.size())
    , n_face_q_points(face_quadrature_in.size())
    , n_dofs(solid_fe_in.n_dofs_per_cell())
    , cell_diameter(0.0)

    // Cell data
    , JxW(n_q_points)
    , quadrature_points(n_q_points)
    , components(n_dofs)

    , base_index(n_dofs)
    , local_dof_indices(n_dofs)
    , alpha_at_vertex(n_dofs, 0.0)
    , picard_alpha_nodal(n_dofs, 0.0)
    , delta_at_vertex(n_dofs, 0.0)
    , fct_delta_nodal(n_dofs, 0.0)
    , fct_delta_gradients(n_q_points)

    , phi_u(n_q_points, std::vector<Tensor<1, dim>>(n_dofs))
    , grad_phi_u(n_q_points, std::vector<Tensor<2, dim>>(n_dofs))
    , hessian_phi_u(n_q_points, std::vector<Tensor<3, dim>>(n_dofs))

    , phi_a(n_q_points, std::vector<double>(n_dofs))
    , grad_phi_a(n_q_points, std::vector<Tensor<1, dim>>(n_dofs))

    , old_velocity_values(n_q_points)
    , old_velocity_gradients(n_q_points)
    , old_alpha_values(n_q_points)
    , old_alpha_gradients(n_q_points)

    , older_velocity_values(n_q_points)
    , older_alpha_values(n_q_points)

    , picard_velocity_values(n_q_points)
    , picard_velocity_gradients(n_q_points)
    , picard_velocity_hessians(n_q_points)
    , picard_velocity_divergences(n_q_points, 0.0)
    , picard_alpha_values(n_q_points)
    , picard_alpha_gradients(n_q_points)
    , picard_theta_values(n_q_points)
    , picard_theta_gradients(n_q_points)

    , fluid_velocity_values(n_q_points)
    , fluid_velocity_gradients(n_q_points)
    , fluid_velocity_divergences(n_q_points, 0.0)
    , fluid_pressure_gradients(n_q_points)

    // Face data
    , face_JxW(n_face_q_points)
    , face_quadrature_points(n_face_q_points)
    , face_normals(n_face_q_points)

    , face_picard_velocity_values(n_face_q_points)
    , face_picard_alpha_values(n_face_q_points)

    , face_phi_u(n_face_q_points, std::vector<Tensor<1, dim>>(n_dofs))
    , face_phi_a(n_face_q_points, std::vector<double>(n_dofs))
  {
    for (unsigned int i = 0; i < n_dofs; ++i)
      {
        components[i] = solid_fe_in.system_to_component_index(i).first;
        base_index[i] = solid_fe_in.system_to_component_index(i).second;
      }
  }

  SolidScratchData(const SolidScratchData<dim> &other)
    : solid_fe(other.solid_fe)
    , theta_fe(other.theta_fe)
    , cell_quadrature(other.cell_quadrature)
    , face_quadrature(other.face_quadrature)
    , fluid_mapping(other.fluid_mapping)
    , fluid_fe(other.fluid_fe)
    , fe_values(*solid_fe,
                *cell_quadrature,
                update_values | update_gradients | update_hessians |
                  update_quadrature_points | update_JxW_values)
    , fe_face_values(*solid_fe,
                     *face_quadrature,
                     update_values | update_normal_vectors |
                       update_quadrature_points | update_JxW_values)
    , theta_fe_values(*theta_fe,
                      *cell_quadrature,
                      update_values | update_gradients)
    , fluid_fe_values(*fluid_mapping,
                      *fluid_fe,
                      *cell_quadrature,
                      update_values | update_gradients)
    , n_q_points(other.n_q_points)
    , n_face_q_points(other.n_face_q_points)
    , n_dofs(other.n_dofs)
    , cell_diameter(0.0)

    // Cell data
    , JxW(n_q_points)
    , quadrature_points(n_q_points)
    , components(other.components)
    , base_index(other.base_index)
    , local_dof_indices(other.n_dofs)
    , alpha_at_vertex(other.n_dofs, 0.0)
    , picard_alpha_nodal(other.n_dofs, 0.0)
    , delta_at_vertex(other.n_dofs, 0.0)
    , fct_delta_nodal(other.n_dofs, 0.0)
    , fct_delta_gradients(other.n_q_points)

    , phi_u(n_q_points, std::vector<Tensor<1, dim>>(n_dofs))
    , grad_phi_u(n_q_points, std::vector<Tensor<2, dim>>(n_dofs))
    , hessian_phi_u(n_q_points, std::vector<Tensor<3, dim>>(n_dofs))


    , phi_a(n_q_points, std::vector<double>(n_dofs))
    , grad_phi_a(n_q_points, std::vector<Tensor<1, dim>>(n_dofs))

    , old_velocity_values(n_q_points)
    , old_velocity_gradients(n_q_points)
    , old_alpha_values(n_q_points)
    , old_alpha_gradients(n_q_points)

    , older_velocity_values(n_q_points)
    , older_alpha_values(n_q_points)

    , picard_velocity_values(n_q_points)
    , picard_velocity_gradients(n_q_points)
    , picard_velocity_hessians(n_q_points)
    , picard_velocity_divergences(n_q_points, 0.0)
    , picard_alpha_values(n_q_points)
    , picard_alpha_gradients(n_q_points)
    , picard_theta_values(n_q_points)
    , picard_theta_gradients(n_q_points)

    , fluid_velocity_values(n_q_points)
    , fluid_velocity_gradients(n_q_points)
    , fluid_velocity_divergences(n_q_points, 0.0)
    , fluid_pressure_gradients(n_q_points)

    // Face data
    , face_JxW(n_face_q_points)
    , face_quadrature_points(n_face_q_points)
    , face_normals(n_face_q_points)

    , face_picard_velocity_values(n_face_q_points)
    , face_picard_alpha_values(n_face_q_points)

    , face_phi_u(n_face_q_points, std::vector<Tensor<1, dim>>(n_dofs))
    , face_phi_a(n_face_q_points, std::vector<double>(n_dofs))
  {}

  void
  reinit(const typename DoFHandler<dim>::active_cell_iterator &cell,
         const typename DoFHandler<dim>::active_cell_iterator &theta_cell,
         const typename DoFHandler<dim>::active_cell_iterator &fluid_cell,
         const TrilinosWrappers::MPI::BlockVector             &old_solution,
         const TrilinosWrappers::MPI::BlockVector             &older_solution,
         const TrilinosWrappers::MPI::BlockVector             &picard_solution,
         const TrilinosWrappers::MPI::BlockVector             &fct_delta,
         const TrilinosWrappers::MPI::Vector                  &theta_solution,
         const TrilinosWrappers::MPI::Vector                  &fluid_solution,
         const FEValuesExtractors::Vector                     &velocities,
         const FEValuesExtractors::Scalar                     &alpha,
         const FEValuesExtractors::Vector                     &fluid_velocities,
         const FEValuesExtractors::Scalar                     &fluid_pressure)
  {
    fe_values.reinit(cell);
    theta_fe_values.reinit(theta_cell);
    fluid_fe_values.reinit(fluid_cell);
    cell_diameter = cell->diameter();

    // Old solution: U^n
    fe_values[velocities].get_function_values(old_solution,
                                              old_velocity_values);
    fe_values[velocities].get_function_gradients(old_solution,
                                                 old_velocity_gradients);

    fe_values[alpha].get_function_values(old_solution, old_alpha_values);
    fe_values[alpha].get_function_gradients(old_solution, old_alpha_gradients);

    // Older solution: U^{n-1}
    fe_values[velocities].get_function_values(older_solution,
                                              older_velocity_values);
    fe_values[alpha].get_function_values(older_solution, older_alpha_values);

    // Picard solution: U^k
    fe_values[velocities].get_function_values(picard_solution,
                                              picard_velocity_values);
    fe_values[velocities].get_function_gradients(picard_solution,
                                                 picard_velocity_gradients);

    fe_values[velocities].get_function_hessians(picard_solution,
                                                picard_velocity_hessians);

    fe_values[alpha].get_function_values(picard_solution, picard_alpha_values);

    fe_values[alpha].get_function_gradients(picard_solution,
                                            picard_alpha_gradients);

    cell->get_dof_indices(local_dof_indices);

    for (unsigned int i = 0; i < n_dofs; ++i)
      if (components[i] == dim)
        alpha_at_vertex[base_index[i]] = picard_solution(local_dof_indices[i]);

    for (unsigned int i = 0; i < n_dofs; ++i)
      picard_alpha_nodal[i] = alpha_at_vertex[base_index[i]];

    // [REVIEW FIX 5] lagged FCT correction delta = alpha^F - alpha^L
    // (nodal values and gradient at the quadrature points)
    for (unsigned int i = 0; i < n_dofs; ++i)
      if (components[i] == dim)
        delta_at_vertex[base_index[i]] = fct_delta(local_dof_indices[i]);

    for (unsigned int i = 0; i < n_dofs; ++i)
      fct_delta_nodal[i] = delta_at_vertex[base_index[i]];

    fe_values[alpha].get_function_gradients(fct_delta, fct_delta_gradients);


    theta_fe_values.get_function_values(theta_solution, picard_theta_values);
    theta_fe_values.get_function_gradients(theta_solution,
                                           picard_theta_gradients);



    // Fluid velocity and pressure fields
    fluid_fe_values[fluid_velocities].get_function_values(
      fluid_solution, fluid_velocity_values);

    fluid_fe_values[fluid_velocities].get_function_gradients(
      fluid_solution, fluid_velocity_gradients);

    fluid_fe_values[fluid_pressure].get_function_gradients(
      fluid_solution, fluid_pressure_gradients);

    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        JxW[q]               = fe_values.JxW(q);
        quadrature_points[q] = fe_values.quadrature_point(q);

        picard_velocity_divergences[q] = trace(picard_velocity_gradients[q]);

        fluid_velocity_divergences[q] = trace(fluid_velocity_gradients[q]);

        for (unsigned int i = 0; i < n_dofs; ++i)
          {
            phi_u[q][i]         = fe_values[velocities].value(i, q);
            grad_phi_u[q][i]    = fe_values[velocities].gradient(i, q);
            hessian_phi_u[q][i] = fe_values[velocities].hessian(i, q);

            phi_a[q][i]      = fe_values[alpha].value(i, q);
            grad_phi_a[q][i] = fe_values[alpha].gradient(i, q);
          }
      }
  }

  void
  reinit_face(const typename DoFHandler<dim>::active_cell_iterator &cell,
              const unsigned int                                    face,
              const TrilinosWrappers::MPI::BlockVector &picard_solution,
              const FEValuesExtractors::Vector         &velocities,
              const FEValuesExtractors::Scalar         &alpha)
  {
    fe_face_values.reinit(cell, face);

    fe_face_values[velocities].get_function_values(picard_solution,
                                                   face_picard_velocity_values);

    fe_face_values[alpha].get_function_values(picard_solution,
                                              face_picard_alpha_values);

    for (unsigned int q = 0; q < n_face_q_points; ++q)
      {
        face_JxW[q]               = fe_face_values.JxW(q);
        face_quadrature_points[q] = fe_face_values.quadrature_point(q);
        face_normals[q]           = fe_face_values.normal_vector(q);

        for (unsigned int i = 0; i < n_dofs; ++i)
          {
            face_phi_u[q][i] = fe_face_values[velocities].value(i, q);

            face_phi_a[q][i] = fe_face_values[alpha].value(i, q);
          }
      }
  }

  const FiniteElement<dim>  *solid_fe;
  const FiniteElement<dim>  *theta_fe;
  const Quadrature<dim>     *cell_quadrature;
  const Quadrature<dim - 1> *face_quadrature;
  const Mapping<dim>        *fluid_mapping;
  const FiniteElement<dim>  *fluid_fe;

  FEValues<dim>     fe_values;
  FEFaceValues<dim> fe_face_values;
  FEValues<dim>     theta_fe_values;
  FEValues<dim>     fluid_fe_values;

  unsigned int n_q_points;
  unsigned int n_face_q_points;
  unsigned int n_dofs;
  double       cell_diameter;

  // Cell data
  std::vector<double>     JxW;
  std::vector<Point<dim>> quadrature_points;



  std::vector<unsigned int> components;

  std::vector<unsigned int>            base_index; // vertex of local dof i
  std::vector<types::global_dof_index> local_dof_indices;
  std::vector<double>                  alpha_at_vertex;
  std::vector<double> picard_alpha_nodal; // alpha^k at the vertex of dof i

  // lagged FCT correction delta = alpha^F - alpha^L (previous Picard iterate)
  std::vector<double>         delta_at_vertex;
  std::vector<double>         fct_delta_nodal; // delta at the vertex of dof i
  std::vector<Tensor<1, dim>> fct_delta_gradients;

  std::vector<std::vector<Tensor<1, dim>>> phi_u;
  std::vector<std::vector<Tensor<2, dim>>> grad_phi_u;
  std::vector<std::vector<Tensor<3, dim>>> hessian_phi_u;

  std::vector<std::vector<double>>         phi_a;
  std::vector<std::vector<Tensor<1, dim>>> grad_phi_a;

  std::vector<Tensor<1, dim>> old_velocity_values;
  std::vector<Tensor<2, dim>> old_velocity_gradients;

  std::vector<double>         old_alpha_values;
  std::vector<Tensor<1, dim>> old_alpha_gradients;

  std::vector<Tensor<1, dim>> older_velocity_values;
  std::vector<double>         older_alpha_values;

  std::vector<Tensor<1, dim>> picard_velocity_values;
  std::vector<Tensor<2, dim>> picard_velocity_gradients;
  std::vector<Tensor<3, dim>> picard_velocity_hessians;
  std::vector<double>         picard_velocity_divergences;
  std::vector<double>         picard_alpha_values;
  std::vector<Tensor<1, dim>> picard_alpha_gradients;
  std::vector<double>         picard_theta_values;
  std::vector<Tensor<1, dim>> picard_theta_gradients;

  std::vector<Tensor<1, dim>> fluid_velocity_values;
  std::vector<Tensor<2, dim>> fluid_velocity_gradients;
  std::vector<double>         fluid_velocity_divergences;
  std::vector<Tensor<1, dim>> fluid_pressure_gradients;

  // Face data
  std::vector<double>         face_JxW;
  std::vector<Point<dim>>     face_quadrature_points;
  std::vector<Tensor<1, dim>> face_normals;

  std::vector<Tensor<1, dim>> face_picard_velocity_values;
  std::vector<double>         face_picard_alpha_values;

  std::vector<std::vector<Tensor<1, dim>>> face_phi_u;
  std::vector<std::vector<double>>         face_phi_a;
};



template <int dim>
class SolidAssemblerBDF : public SolidTimeAssemblerBase<dim>
{
public:
  SolidAssemblerBDF(
    const std::shared_ptr<SimulationControl> &simulation_control,
    const SolidPhaseParameters<dim>          &parameters)
    : simulation_control(simulation_control)
    , rho_s(parameters.rho_s)
    , dt(simulation_control->get_time_step())
    , assembly_method(Parameters::SimulationControl::TimeSteppingMethod::bdf1)
  {}

  void
  update_coefficients(const double /*unused*/) override
  {
    AssertThrow(simulation_control != nullptr,
                ExcMessage("simulation_control is null in SolidAssemblerBDF."));

    dt              = simulation_control->get_time_step();
    assembly_method = simulation_control->get_assembly_method();
  }

  void
  assemble(const SolidScratchData<dim> &scratch,
           SolidCopyData<dim>          &copy) const override
  {
    switch (assembly_method)
      {
        case Parameters::SimulationControl::TimeSteppingMethod::steady:
          return;

        case Parameters::SimulationControl::TimeSteppingMethod::steady_bdf:
        case Parameters::SimulationControl::TimeSteppingMethod::bdf1:
          assemble_bdf1(scratch, copy);
          return;

        case Parameters::SimulationControl::TimeSteppingMethod::bdf2:
          assemble_bdf2(scratch, copy);
          return;

        default:
          AssertThrow(
            false,
            ExcMessage(
              "Unsupported time stepping method in SolidAssemblerBDF."));
      }
  }

private:
  void
  assemble_bdf1(const SolidScratchData<dim> &scratch,
                SolidCopyData<dim>          &copy) const
  {
    const unsigned int n_q   = scratch.n_q_points;
    const unsigned int n_dof = scratch.n_dofs;

    for (unsigned int q = 0; q < n_q; ++q)
      {
        const double          JxW   = scratch.JxW[q];
        const Tensor<1, dim> &u_old = scratch.old_velocity_values[q];
        const double          a_old = scratch.old_alpha_values[q];


        const double a_pic = scratch.picard_alpha_values[q];


        for (unsigned int i = 0; i < n_dof; ++i)
          {
            const unsigned int comp_i = scratch.components[i];

            // --------------------------------------------------
            // continuity BDF1 time term:

            // --------------------------------------------------
            if (comp_i == dim)
              {
                const double q_i = scratch.phi_a[q][i];

                // --------------------------------------------------
                // NEW: lumped mass for low-order alpha equation
                // --------------------------------------------------
                //
                // m_i = sum_j M_ij
                //     = integral phi_i * sum_j(phi_j)
                //     = integral phi_i
                //
                // because sum_j phi_j = 1.
                //
                copy.local_alpha_low_matrix(i, i) += (q_i / dt) * JxW;

                // [REVIEW FIX 5] The monolithic continuity block now uses
                // the SAME lumped mass as the low-order predictor, so that at
                // Picard convergence alpha^H = alpha^L. The RHS m_i alpha^n_i
                // / dt is added in assemble_local_system (nodal alpha^n).
                copy.local_matrix(i, i) += (q_i / dt) * JxW;

                for (unsigned int j = 0; j < n_dof; ++j)
                  {
                    if (scratch.components[j] != dim)
                      continue;

                    const double alpha_j = scratch.phi_a[q][j];

                    // consistent mass: still needed by the FCT fluxes
                    copy.local_alpha_consistent_mass(i, j) +=
                      q_i * alpha_j * JxW;
                  }

                (void)a_old;
              }

            // --------------------------------------------------
            // Momentum BDF1 time term:

            // --------------------------------------------------
            //
            else if (comp_i < dim)
              {
                // Phase-intensive momentum: alpha * rho_s * (u - u^n)/dt,
                // with alpha frozen at the Picard value of the node of v_i.
                constexpr double alpha_min_momentum = 1e-3;

                const Tensor<1, dim> &v_i = scratch.phi_u[q][i];

                const double a_i =
                  std::max(scratch.picard_alpha_nodal[i], alpha_min_momentum);

                for (unsigned int j = 0; j < n_dof; ++j)
                  {
                    if (scratch.components[j] != comp_i)
                      continue;

                    const Tensor<1, dim> &u_j = scratch.phi_u[q][j];

                    copy.local_matrix(i, j) +=
                      (rho_s * a_i / dt) * (v_i * u_j) * JxW;
                  }

                copy.local_rhs(i) += (rho_s * a_i / dt) * (v_i * u_old) * JxW;
              }
          }
      }
  }

  // need to check, not fully ready
  void
  assemble_bdf2(const SolidScratchData<dim> &scratch,
                SolidCopyData<dim>          &copy) const
  {
    const unsigned int n_q   = scratch.n_q_points;
    const unsigned int n_dof = scratch.n_dofs;

    for (unsigned int q = 0; q < n_q; ++q)
      {
        const double          JxW     = scratch.JxW[q];
        const double          a_old   = scratch.old_alpha_values[q];
        const double          a_older = scratch.older_alpha_values[q];
        const Tensor<1, dim> &u_old   = scratch.old_velocity_values[q];
        const Tensor<1, dim> &u_older = scratch.older_velocity_values[q];

        for (unsigned int i = 0; i < n_dof; ++i)
          {
            const unsigned int comp_i = scratch.components[i];

            // ----------------
            // continuity equation
            // ----------------
            if (comp_i == dim)
              {
                const double w_i = scratch.phi_a[q][i];

                // implicit BDF2 mass: (3/2)/dt
                for (unsigned int j = 0; j < n_dof; ++j)
                  {
                    if (scratch.components[j] != dim)
                      continue;

                    const double a_j = scratch.phi_a[q][j];
                    copy.local_matrix(i, j) += (1.5 / dt) * (w_i * a_j) * JxW;
                  }

                // RHS: (2 a^n - 0.5 a^{n-1}) / dt
                const double rhs_alpha = (2.0 * a_old - 0.5 * a_older) / dt;
                copy.local_rhs(i) += w_i * rhs_alpha * JxW;
              }

            // ----------------
            // Momentum equation
            // ----------------
            else if (comp_i < dim)
              {
                const Tensor<1, dim> &phi_u_i = scratch.phi_u[q][i];

                for (unsigned int j = 0; j < n_dof; ++j)
                  {
                    if (scratch.components[j] != comp_i)
                      continue;

                    const Tensor<1, dim> &phi_u_j = scratch.phi_u[q][j];

                    // implicit mass: (3/2)(rho_s * a^{n+1})/dt
                    copy.local_matrix(i, j) +=
                      (1.5 * rho_s * scratch.picard_alpha_values[q] / dt) *
                      (phi_u_i * phi_u_j) * JxW;
                  }

                // RHS: (rho_s/dt)( 2 a^n u^n  - 0.5 a^{n-1} u^{n-1} )
                Tensor<1, dim> momentum_bdf_rhs =
                  (2.0 * a_old * u_old - 0.5 * a_older * u_older) / dt;

                copy.local_rhs(i) += (phi_u_i * momentum_bdf_rhs) * JxW;
              }
          }
      }
  }

  std::shared_ptr<SimulationControl>                simulation_control;
  const double                                      rho_s;
  double                                            dt;
  Parameters::SimulationControl::TimeSteppingMethod assembly_method;
};



template <int dim>
Tensor<1, dim>
divergence_solid_stress_frozen_coefficients(
  const double          mu_s,
  const double          lambda_s,
  const Tensor<3, dim> &velocity_hessian)
{
  Tensor<1, dim> laplacian_velocity;
  Tensor<1, dim> gradient_divergence;

  // (Delta u)_c = sum_d d^2 u_c / dx_d^2
  for (unsigned int c = 0; c < dim; ++c)
    for (unsigned int d = 0; d < dim; ++d)
      laplacian_velocity[c] += velocity_hessian[c][d][d];

  // [grad(div u)]_d = sum_c d^2 u_c / (dx_d dx_c)
  for (unsigned int d = 0; d < dim; ++d)
    for (unsigned int c = 0; c < dim; ++c)
      gradient_divergence[d] += velocity_hessian[c][d][c];

  // tau = mu(grad u + grad u^T)
  //       + (lambda - 2 mu/3) div(u) I
  //
  // With locally frozen mu and lambda:
  // div(tau) = mu Delta(u) + (lambda + mu/3) grad(div u).
  return mu_s * laplacian_velocity +
         (lambda_s + mu_s / 3.0) * gradient_divergence;
}


template <int dim>
class SolidCoreAssembler
{
public:
  explicit SolidCoreAssembler(
    const SolidPhaseParameters<dim>          &parameters,
    const std::shared_ptr<SimulationControl> &simulation_control)
    : parameters(parameters)
    , simulation_control(simulation_control)
    , rho_s(parameters.rho_s)
    , beta(parameters.beta)
  {}

  void
  assemble_cell(const SolidScratchData<dim> &scratch,
                SolidCopyData<dim>          &copy) const
  {
    const unsigned int n_q   = scratch.n_q_points;
    const unsigned int n_dof = scratch.n_dofs;

    auto &A = copy.local_matrix;
    auto &F = copy.local_rhs;

    const double dt = simulation_control->get_time_step();

    // SUPG controls
    const bool   use_alpha_supg       = parameters.use_alpha_supg;
    const double alpha_supg_factor    = parameters.alpha_supg_factor;
    const bool   use_momentum_supg    = parameters.use_momentum_supg;
    const double momentum_supg_factor = parameters.momentum_supg_factor;

    const double tiny_velocity      = 1e-12;
    const double tiny_cell_diameter = 1e-12;


    const bool   use_solid_grad_div = parameters.use_solid_grad_div;
    const double grad_div_factor    = parameters.grad_div_factor;

    // AssertThrow(!use_momentum_supg,
    //         ExcMessage("Momentum SUPG is not consistent with the "
    //                     "phase-intensive solid momentum. "
    //                     "Set 'use momentum supg = false'."));

    constexpr double alpha_min_momentum = 1e-3;

    // Johnson-Jackson frictional pressure and its derivative at the
    // vertices (group finite element formulation). p_f depends on alpha
    // only; theta is passed only because the function signature needs it.
    std::vector<double> pf_node(n_dof, 0.0);
    std::vector<double> Gf_node(n_dof, 0.0);

    for (unsigned int j = 0; j < n_dof; ++j)
      if (scratch.components[j] == dim)
        {
          // max(.,0) only guards round-off (-1e-20) in a function that is
          // identically zero below alpha_friction; it changes nothing.
          const SolidKTGFValues kj =
            evaluate_solid_ktgf(parameters,
                                std::max(scratch.picard_alpha_nodal[j], 0.0),
                                scratch.picard_theta_values[0],
                                0.0);

          pf_node[j] = kj.frictional_pressure;
          Gf_node[j] = kj.frictional_pressure_derivative;
        }

    // [REVIEW FIX 6] Momentum convection rho_s a_i (grad u) u^k as a scalar
    // vertex matrix (identical for every velocity component); discrete
    // upwinding is applied after the quadrature loop. Galerkin convection
    // with tau_u ~ dt (CFL ~ 0.01) had no effective stabilization and gave
    // the node-to-node velocity zig-zag in the freeboard.
    const unsigned int n_vertices = n_dof / (dim + 1);
    FullMatrix<double> K_conv(n_vertices, n_vertices);


    for (unsigned int q = 0; q < n_q; ++q)
      {
        const double JxW = scratch.JxW[q];

        Vector<double> source_values(dim + 1);

        Tensor<1, dim> S_m;
        double         S_alpha = 0.0;

        if (parameters.source_term.enable)
          {
            parameters.source_term.function->vector_value(
              scratch.quadrature_points[q], source_values);

            for (unsigned int d = 0; d < dim; ++d)
              S_m[d] = source_values[d];

            S_alpha = source_values[dim];
          }

        const Tensor<1, dim> &u_pic = scratch.picard_velocity_values[q];
        const double          a_pic = scratch.picard_alpha_values[q];

        const Tensor<1, dim> &u_f = scratch.fluid_velocity_values[q];

        const double theta_pic = scratch.picard_theta_values[q];

        const Tensor<1, dim> &grad_p = scratch.fluid_pressure_gradients[q];

        const Tensor<1, dim> &grad_a_pic = scratch.picard_alpha_gradients[q];

        const Tensor<2, dim> &grad_u_pic = scratch.picard_velocity_gradients[q];

        const Tensor<3, dim> &hessian_u_pic =
          scratch.picard_velocity_hessians[q];

        const double div_u_pic = scratch.picard_velocity_divergences[q];

        const SolidKTGFValues ktgf =
          evaluate_solid_ktgf(parameters, a_pic, theta_pic, div_u_pic);

        const Tensor<2, dim> tau_pic =
          make_solid_stress<dim>(ktgf.shear_viscosity,
                                 ktgf.bulk_viscosity,
                                 grad_u_pic);

        // div(alpha^k u^k)
        const double div_alpha_u_pic = (u_pic * grad_a_pic) + a_pic * div_u_pic;

        // div(tau(u^k)), with locally frozen KTGF coefficients
        const Tensor<1, dim> div_tau_pic =
          divergence_solid_stress_frozen_coefficients<dim>(ktgf.shear_viscosity,
                                                           ktgf.bulk_viscosity,
                                                           hessian_u_pic);

        // div(alpha^k tau(u^k))
        const Tensor<1, dim> div_alpha_tau_pic =
          tau_pic * grad_a_pic + a_pic * div_tau_pic;

        // div(alpha^k u^k tensor u^k)
        const Tensor<1, dim> conservative_convection_pic =
          a_pic * (grad_u_pic * u_pic) + div_alpha_u_pic * u_pic;

        const double h = std::max(scratch.cell_diameter, tiny_cell_diameter);

        const double u_norm = std::max(u_pic.norm(), tiny_velocity);

        const double tau_alpha =
          alpha_supg_factor /
          std::sqrt((1.0 / dt) * (1.0 / dt) +
                    (2.0 * u_norm / h) * (2.0 * u_norm / h));

        const double tau_u = momentum_supg_factor /
                             std::sqrt((1.0 / dt) * (1.0 / dt) +
                                       (2.0 * u_norm / h) * (2.0 * u_norm / h));

        for (unsigned int i = 0; i < n_dof; ++i)
          {
            const unsigned int comp_i = scratch.components[i];

            // --------------------------------------------------
            // Continuity equation
            // --------------------------------------------------
            if (comp_i == dim)
              {
                const double q_i = scratch.phi_a[q][i];

                const Tensor<1, dim> &grad_q_i = scratch.grad_phi_a[q][i];

                const double supg_test_alpha = u_pic * grad_q_i;


                // ----------------------------------------------------------
                // Implicit particle-pressure (packing) diffusion.
                // Passalacqua & Fox 2011, Eq. 50:
                //   D_pp = C (dt / rho_s) * alpha^k * G(alpha^k),
                //   G = dp_f/dalpha  (frictional packing modulus).
                // Implicit in alpha, parabolic => unconditionally stable;
                // D_pp -> large as alpha -> alpha_max, so it enforces
                // alpha <= alpha_max at the source instead of post-hoc FCT.
                // ----------------------------------------------------------

                // [REVIEW FIX 4] OFF. In Passalacqua & Fox this term is
                // consistent only because their momentum flux has the
                // particle-pressure gradient removed. In this monolithic
                // solve grad(p_s) is already implicit in the momentum
                // (A_u_alpha), so D_pp is an extra O(dt) mass flux that is
                // not in the equations. Weighted grad-div (below) replaces
                // it as the dense-phase stabilization.
                const double pp_diffusion_factor = 1.0;
                const double D_pp = pp_diffusion_factor * (dt / rho_s) * a_pic *
                                    (ktgf.frictional_pressure_derivative);

                // --------------------------------------------------
                // A_alpha_alpha
                // --------------------------------------------------
                for (unsigned int j = 0; j < n_dof; ++j)
                  {
                    if (scratch.components[j] != dim)
                      continue;

                    const double alpha_j = scratch.phi_a[q][j];

                    const Tensor<1, dim> &grad_alpha_j =
                      scratch.grad_phi_a[q][j];

                    double val = 0.0;

                    // Conservative Galerkin volume term:
                    // - integral alpha^{k+1} u^k . grad(q)
                    val += -alpha_j * (u_pic * grad_q_i);

                    // Implicit particle-pressure diffusion:
                    //   + integral D_pp grad(alpha_j) . grad(q_i)
                    // Symmetric, negative off-diagonals => M-matrix friendly,
                    // reinforces the discrete maximum principle.
                    val += D_pp * (grad_alpha_j * grad_q_i);

                    copy.local_alpha_convection(i, j) +=
                      D_pp * (grad_alpha_j * grad_q_i) * JxW;


                    // --------------------------------------------------
                    // NEW: raw Galerkin convection matrix for
                    // low-order FCT predictor.
                    //
                    // IMPORTANT:
                    // no SUPG is included here.
                    // --------------------------------------------------
                    copy.local_alpha_convection(i, j) +=
                      -alpha_j * (u_pic * grad_q_i) * JxW;

                    // Full continuity SUPG alpha-column contribution:
                    // tau (u^k.grad q)
                    // [alpha_j/dt + div(alpha_j u^k)]
                    if (use_alpha_supg)
                      {
                        const double residual_alpha_part =
                          alpha_j / dt + (u_pic * grad_alpha_j) +
                          alpha_j * div_u_pic;

                        val +=
                          tau_alpha * supg_test_alpha * residual_alpha_part;
                      }

                    A(i, j) += val * JxW;
                  }

                // --------------------------------------------------
                // A_alpha_u
                // --------------------------------------------------
                for (unsigned int j = 0; j < n_dof; ++j)
                  {
                    if (scratch.components[j] >= dim)
                      continue;

                    const Tensor<1, dim> &u_trial_j = scratch.phi_u[q][j];

                    const Tensor<2, dim> &grad_u_j = scratch.grad_phi_u[q][j];

                    // Galerkin Picard coupling:
                    // - integral alpha^k u_j . grad(q)
                    A(i, j) += -a_pic * (u_trial_j * grad_q_i) * JxW;

                    // Full continuity SUPG velocity-column contribution:
                    // tau (u^k.grad q) div(alpha^k u_j)
                    if (use_alpha_supg)
                      {
                        const double residual_velocity_part =
                          (grad_a_pic * u_trial_j) + a_pic * trace(grad_u_j);

                        A(i, j) += tau_alpha * supg_test_alpha *
                                   residual_velocity_part * JxW;
                      }
                  }

                // --------------------------------------------------
                // Galerkin Picard correction
                // --------------------------------------------------
                F(i) += -a_pic * (u_pic * grad_q_i) * JxW;

                // --------------------------------------------------
                // Full continuity SUPG RHS
                // --------------------------------------------------
                if (use_alpha_supg)
                  {
                    double residual_rhs_part =
                      scratch.old_alpha_values[q] / dt + div_alpha_u_pic;

                    if (parameters.source_term.enable)
                      residual_rhs_part += S_alpha;

                    F(i) +=
                      tau_alpha * supg_test_alpha * residual_rhs_part * JxW;
                  }

                if (parameters.source_term.enable)
                  F(i) += q_i * S_alpha * JxW;
              }

            //

            // --------------------------------------------------
            // Momentum equation, phase-intensive form:
            //
            //   a_i [ rho_s (du/dt + (grad u) u) + beta (u - u_f)
            //         + grad p - rho_s g ]
            //   + grad p_kin + grad p_f,h - div(alpha tau) + C_p grad Theta = 0
            //
            // a_i = max(alpha^k at the node of v_i, alpha_min_momentum).
            // Identical to the conservative form for the exact solution; its
            // discrete version stays well posed as alpha -> 0.
            // p_f,h = sum_j p_f(alpha_j) phi_j (nodal), Newton-linearised.
            // --------------------------------------------------
            else if (comp_i < dim)
              {
                const Tensor<1, dim> &v_i      = scratch.phi_u[q][i];
                const Tensor<2, dim> &grad_v_i = scratch.grad_phi_u[q][i];

                const double a_i =
                  std::max(scratch.picard_alpha_nodal[i], alpha_min_momentum);

                // Dilute limit: no particle continuum (mean free path ~ domain)
                // -> particle stresses fade out; the node follows drag +
                // gravity + gas pressure.
                constexpr double alpha_dilute = 1e-3;
                const double     w_i =
                  std::clamp(scratch.picard_alpha_nodal[i] / alpha_dilute,
                             0.0,
                             1.0);

                // Stress force weighted like this row's inertia (per-particle
                // form): (a_i / alpha_q) div(sigma), capped at 1, times the
                // dilute cutoff w_i.
                const double s_iq =
                  w_i *
                  std::min(1.0, a_i / std::max(a_pic, alpha_min_momentum));

                // ---------------- A_uu ----------------
                for (unsigned int j = 0; j < n_dof; ++j)
                  {
                    if (scratch.components[j] >= dim)
                      continue;

                    const Tensor<1, dim> &u_trial_j = scratch.phi_u[q][j];
                    const Tensor<2, dim> &grad_u_j  = scratch.grad_phi_u[q][j];

                    double val = 0.0;

                    // Non-conservative convection (Picard): a_i rho_s (grad
                    // u_j) u^k -> stored once (component 0) in K_conv and
                    // added after upwinding, see [REVIEW FIX 6].
                    if (comp_i == 0 && scratch.components[j] == 0)
                      K_conv(scratch.base_index[i], scratch.base_index[j]) +=
                        rho_s * a_i * (v_i * (grad_u_j * u_pic)) * JxW;

                    // Drag
                    val += beta * a_i * (v_i * u_trial_j);

                    // Solid stress: (alpha^k tau(u_j)) : grad v_i
                    const Tensor<2, dim> tau_trial =
                      make_solid_stress<dim>(ktgf.shear_viscosity,
                                             ktgf.bulk_viscosity,
                                             grad_u_j);

                    val += s_iq * a_pic * scalar_product(tau_trial, grad_v_i);

                    A(i, j) += val * JxW;
                  }

                // ---------------- A_u_alpha: particle pressure only
                // ----------------
                for (unsigned int j = 0; j < n_dof; ++j)
                  {
                    if (scratch.components[j] != dim)
                      continue;

                    const Tensor<1, dim> &grad_alpha_j =
                      scratch.grad_phi_a[q][j];

                    const double v_dot_grad = v_i * grad_alpha_j;

                    // Kinetic part: G_kin(alpha^k) grad alpha
                    A(i, j) += s_iq * ktgf.particle_pressure_derivative *
                               v_dot_grad * JxW;

                    // Frictional part, nodal:
                    // p_f(alpha_j) ~ p_f(alpha_j^k) + G_f(alpha_j^k)(alpha_j -
                    // alpha_j^k)
                    A(i, j) += s_iq * Gf_node[j] * v_dot_grad * JxW;

                    F(i) -= s_iq *
                            (pf_node[j] -
                             Gf_node[j] * scratch.picard_alpha_nodal[j]) *
                            v_dot_grad * JxW;

                    // [REVIEW FIX 5] lagged FCT correction, friction part:
                    // the momentum sees alpha^H + delta, which equals the
                    // stored alpha^F at Picard convergence.
                    F(i) -= s_iq * Gf_node[j] * scratch.fct_delta_nodal[j] *
                            v_dot_grad * JxW;
                  }

                // [REVIEW FIX 5] lagged FCT correction, kinetic part
                F(i) -= s_iq * ktgf.particle_pressure_derivative *
                        (v_i * scratch.fct_delta_gradients[q]) * JxW;

                // ---------------- RHS ----------------
                // Drag, gas-velocity part
                F(i) += beta * a_i * (v_i * u_f) * JxW;

                // Shared fluid pressure
                // F(i) -= a_i * (v_i * grad_p) * JxW;
                F(i) -= a_i * parameters.rho_f * (v_i * grad_p) * JxW;

                // Gravity
                F(i) += rho_s * a_i * (v_i * parameters.gravity) * JxW;

                // Kinetic granular pressure, Theta part: C_p grad Theta
                const double e_p_local = parameters.restitution_coefficient;
                const double C_p =
                  rho_s * a_pic *
                  (1.0 +
                   2.0 * a_pic * ktgf.radial_distribution * (1.0 + e_p_local));

                F(i) -=
                  s_iq * C_p * (v_i * scratch.picard_theta_gradients[q]) * JxW;

                // Grad-div / continuity-residual stabilization: your old
                // lines 1953-2025, unchanged. Switched off in the prm
                // (use solid grad-div = false).
                if (use_solid_grad_div)
                  {
                    const double div_v_i = trace(grad_v_i);

                    const double h =
                      std::max(scratch.cell_diameter, tiny_cell_diameter);

                    const double u_norm = std::max(u_pic.norm(), tiny_velocity);

                    // const double gamma_gd = grad_div_factor * u_norm * h;
                    // const double gamma_gd = grad_div_factor * rho_s * a_pic *
                    // h * h / dt;

                    const double u_ref =
                      std::sqrt(parameters.gravity.norm() * h);

                    const double gamma_gd =
                      grad_div_factor * std::max(u_norm, u_ref) * h;


                    const Tensor<1, dim> &grad_a_pic =
                      scratch.picard_alpha_gradients[q];

                    const double div_u_pic =
                      scratch.picard_velocity_divergences[q];

                    const double div_alpha_u_pic =
                      (u_pic * grad_a_pic) + a_pic * div_u_pic;

                    // Velocity columns
                    for (unsigned int j = 0; j < n_dof; ++j)
                      {
                        if (scratch.components[j] >= dim)
                          continue;

                        const Tensor<1, dim> &u_trial_j = scratch.phi_u[q][j];

                        const Tensor<2, dim> &grad_u_j =
                          scratch.grad_phi_u[q][j];

                        const double div_u_j = trace(grad_u_j);

                        const double residual_u_part =
                          (grad_a_pic * u_trial_j) + a_pic * div_u_j;

                        // [REVIEW FIX 7] weighted like the stresses (s_iq);
                        // unweighted it dominated the near-empty rows
                        // (a_i = 1e-3) next to the bed and blew them up.
                        A(i, j) += s_iq * rho_s * gamma_gd * residual_u_part *
                                   div_v_i * JxW;
                      }

                    // Alpha columns
                    for (unsigned int j = 0; j < n_dof; ++j)
                      {
                        if (scratch.components[j] != dim)
                          continue;

                        const double alpha_j = scratch.phi_a[q][j];

                        const Tensor<1, dim> &grad_alpha_j =
                          scratch.grad_phi_a[q][j];

                        const double residual_alpha_part =
                          alpha_j / dt + (u_pic * grad_alpha_j) +
                          alpha_j * div_u_pic;

                        A(i, j) += s_iq * rho_s * gamma_gd *
                                   residual_alpha_part * div_v_i * JxW;
                      }

                    // RHS contribution
                    F(i) +=
                      s_iq * rho_s * gamma_gd *
                      ((scratch.old_alpha_values[q] / dt) + div_alpha_u_pic) *
                      div_v_i * JxW;
                  }

                // ---------------- SUPG, consistent with the row-scaled
                // equation ----
                if (use_momentum_supg)
                  {
                    const Tensor<1, dim> supg_test_u =
                      grad_v_i * u_pic; // (grad v_i) u^k

                    // velocity columns
                    for (unsigned int j = 0; j < n_dof; ++j)
                      {
                        if (scratch.components[j] >= dim)
                          continue;

                        const Tensor<1, dim> &u_trial_j = scratch.phi_u[q][j];
                        const Tensor<2, dim> &grad_u_j =
                          scratch.grad_phi_u[q][j];

                        const Tensor<2, dim> tau_trial_s =
                          make_solid_stress<dim>(ktgf.shear_viscosity,
                                                 ktgf.bulk_viscosity,
                                                 grad_u_j);
                        const Tensor<1, dim> div_tau_trial_s =
                          divergence_solid_stress_frozen_coefficients<dim>(
                            ktgf.shear_viscosity,
                            ktgf.bulk_viscosity,
                            scratch.hessian_phi_u[q][j]);

                        Tensor<1, dim> r =
                          a_i * ((rho_s / dt) * u_trial_j +
                                 rho_s * (grad_u_j * u_pic) + beta * u_trial_j);
                        r -= s_iq * (tau_trial_s * grad_a_pic +
                                     a_pic * div_tau_trial_s);

                        A(i, j) += tau_u * (supg_test_u * r) * JxW;
                      }

                    // alpha columns: particle pressure (kinetic + nodal
                    // friction)
                    Tensor<1, dim>
                      friction_const; // Newton constant part of grad p_f,h
                    for (unsigned int j = 0; j < n_dof; ++j)
                      {
                        if (scratch.components[j] != dim)
                          continue;

                        const Tensor<1, dim> &grad_alpha_j =
                          scratch.grad_phi_a[q][j];

                        A(i, j) +=
                          tau_u * s_iq *
                          (ktgf.particle_pressure_derivative + Gf_node[j]) *
                          (supg_test_u * grad_alpha_j) * JxW;

                        friction_const +=
                          (pf_node[j] -
                           Gf_node[j] * scratch.picard_alpha_nodal[j]) *
                          grad_alpha_j;
                      }

                    // right-hand side
                    Tensor<1, dim> r_rhs =
                      a_i * ((rho_s / dt) * scratch.old_velocity_values[q] +
                             beta * u_f - parameters.rho_f * grad_p +
                             rho_s * parameters.gravity);
                    r_rhs -= s_iq * (C_p * scratch.picard_theta_gradients[q] +
                                     friction_const);

                    F(i) += tau_u * (supg_test_u * r_rhs) * JxW;
                  }


                if (parameters.source_term.enable)
                  F(i) += v_i * S_m * JxW;
              }
          }
      }

    // [REVIEW FIX 6] discrete upwinding of the momentum convection
    // (same algebra as the low-order alpha operator, K on the LHS):
    //   d_IJ = max(0, K_IJ, K_JI),  K_IJ -= d,  K_JI -= d,
    //   K_II += d,  K_JJ += d      (row sums unchanged)
    for (unsigned int I = 0; I < n_vertices; ++I)
      for (unsigned int J = I + 1; J < n_vertices; ++J)
        {
          const double d = std::max({0.0, K_conv(I, J), K_conv(J, I)});

          K_conv(I, J) -= d;
          K_conv(J, I) -= d;
          K_conv(I, I) += d;
          K_conv(J, J) += d;
        }

    for (unsigned int i = 0; i < n_dof; ++i)
      {
        const unsigned int comp_i = scratch.components[i];
        if (comp_i >= dim)
          continue;

        for (unsigned int j = 0; j < n_dof; ++j)
          if (scratch.components[j] == comp_i)
            A(i, j) += K_conv(scratch.base_index[i], scratch.base_index[j]);
      }
  }


  void
  assemble_boundary_face(const SolidScratchData<dim> &scratch,
                         const types::boundary_id     boundary_id,
                         SolidCopyData<dim>          &copy) const
  {
    auto &A = copy.local_matrix;

    const unsigned int n_face_q = scratch.n_face_q_points;
    const unsigned int n_dof    = scratch.n_dofs;

    if (!is_outlet_boundary(boundary_id))
      return;

    for (unsigned int q = 0; q < n_face_q; ++q)
      {
        const double JxW = scratch.face_JxW[q];

        const Tensor<1, dim> &normal = scratch.face_normals[q];

        const Tensor<1, dim> &u_pic = scratch.face_picard_velocity_values[q];

        const double a_pic = scratch.face_picard_alpha_values[q];

        const double un = u_pic * normal;

        for (unsigned int i = 0; i < n_dof; ++i)
          {
            const unsigned int comp_i = scratch.components[i];

            // Continuity boundary flux
            if (comp_i == dim)
              {
                if (un <= 0.0)
                  continue;

                const double q_i = scratch.face_phi_a[q][i];

                for (unsigned int j = 0; j < n_dof; ++j)
                  {
                    if (scratch.components[j] != dim)
                      continue;

                    const double alpha_j = scratch.face_phi_a[q][j];

                    A(i, j) += q_i * alpha_j * un * JxW;

                    copy.local_alpha_convection(i, j) +=
                      q_i * alpha_j * un * JxW;
                  }

                // A_alpha_u
                for (unsigned int j = 0; j < n_dof; ++j)
                  {
                    if (scratch.components[j] >= dim)
                      continue;

                    const Tensor<1, dim> &u_trial_j = scratch.face_phi_u[q][j];

                    A(i, j) += q_i * a_pic * (u_trial_j * normal) * JxW;
                  }

                // Picard correction
                copy.local_rhs(i) += q_i * a_pic * un * JxW;
              }

            // // Momentum boundary flux
            // else if (comp_i < dim)
            //   {
            //     if (un <= 0.0)
            //       continue;

            //     const Tensor<1, dim> &v_i = scratch.face_phi_u[q][i];

            //     // Velocity columns
            //     for (unsigned int j = 0; j < n_dof; ++j)
            //       {
            //         if (scratch.components[j] >= dim)
            //           continue;

            //         const Tensor<1, dim> &u_trial_j =
            //         scratch.face_phi_u[q][j];

            //         A(i, j) += rho_s * a_pic * un * (v_i * u_trial_j) * JxW;
            //       }

            //     // Alpha columns
            //     for (unsigned int j = 0; j < n_dof; ++j)
            //       {
            //         if (scratch.components[j] != dim)
            //           continue;

            //         const double alpha_j = scratch.face_phi_a[q][j];

            //         A(i, j) += rho_s * alpha_j * un * (v_i * u_pic) * JxW;
            //       }

            //     // Picard correction
            //     copy.local_rhs(i) += rho_s * a_pic * un * (v_i * u_pic) *
            //     JxW;
            //   }
          }
      }
  }

private:
  BoundaryConditions::BoundaryType
  get_boundary_type(const types::boundary_id boundary_id) const
  {
    const auto it = parameters.boundary_conditions.type.find(boundary_id);

    AssertThrow(it != parameters.boundary_conditions.type.end(),
                ExcMessage(
                  "No solid boundary condition specified for boundary ID " +
                  std::to_string(static_cast<unsigned int>(boundary_id))));

    return it->second;
  }

  bool
  is_inlet_boundary(const types::boundary_id boundary_id) const
  {
    return get_boundary_type(boundary_id) ==
           BoundaryConditions::BoundaryType::function;
  }

  bool
  is_outlet_boundary(const types::boundary_id boundary_id) const
  {
    return get_boundary_type(boundary_id) ==
           BoundaryConditions::BoundaryType::outlet;
  }

  bool
  is_wall_boundary(const types::boundary_id boundary_id) const
  {
    const auto type = get_boundary_type(boundary_id);

    return type == BoundaryConditions::BoundaryType::noslip ||
           type == BoundaryConditions::BoundaryType::slip;
  }

  bool
  is_periodic_boundary(const types::boundary_id boundary_id) const
  {
    return get_boundary_type(boundary_id) ==
           BoundaryConditions::BoundaryType::periodic;
  }

  bool
  is_strong_or_wall_boundary(const types::boundary_id boundary_id) const
  {
    const auto type = get_boundary_type(boundary_id);

    return type == BoundaryConditions::BoundaryType::function ||
           type == BoundaryConditions::BoundaryType::noslip ||
           type == BoundaryConditions::BoundaryType::slip ||
           type == BoundaryConditions::BoundaryType::periodic;
  }

  const SolidPhaseParameters<dim>         &parameters;
  const std::shared_ptr<SimulationControl> simulation_control;
  const double                             rho_s;
  const double                             beta;
};


template <int dim>
void
SolidPhaseSolver<dim>::solid_phase_conservation_monitoring() const
{
  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar alpha(dim);

  const QGauss<dim>     cell_quadrature(order + 1);
  const QGauss<dim - 1> face_quadrature(order + 1);

  FEValues<dim> fe_values(fe,
                          cell_quadrature,
                          update_values | update_gradients | update_JxW_values);

  FEFaceValues<dim> fe_face_values(fe,
                                   face_quadrature,
                                   update_values | update_normal_vectors |
                                     update_JxW_values);

  const unsigned int n_q      = cell_quadrature.size();
  const unsigned int n_face_q = face_quadrature.size();

  std::vector<Tensor<1, dim>> u_values(n_q);
  std::vector<Tensor<2, dim>> grad_u_values(n_q);

  std::vector<double>         alpha_values(n_q);
  std::vector<double>         alpha_old_values(n_q);
  std::vector<Tensor<1, dim>> grad_alpha_values(n_q);

  std::vector<Tensor<1, dim>> face_u_values(n_face_q);
  std::vector<double>         face_alpha_values(n_face_q);

  double local_mass_new = 0.0;
  double local_mass_old = 0.0;
  double local_volume   = 0.0;

  double local_div_min = std::numeric_limits<double>::max();
  double local_div_max = -std::numeric_limits<double>::max();
  double local_div_l2  = 0.0;

  double local_alpha_res_l2  = 0.0;
  double local_alpha_res_min = std::numeric_limits<double>::max();
  double local_alpha_res_max = -std::numeric_limits<double>::max();

  /*
   * computes the volume integral of
   *
   *   d(alpha_s)/dt + div(alpha_s u_s)
   *
   */
  double local_solid_phase_continuity           = 0.0;
  double local_solid_phase_max_local_continuity = 0.0;

  double local_flux_total  = 0.0;
  double local_flux_wall   = 0.0;
  double local_flux_inlet  = 0.0;
  double local_flux_outlet = 0.0;

  double local_outlet_backflow_flux = 0.0;
  double local_outlet_backflow_area = 0.0;

  const double dt = simulation_control->get_time_step();

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      fe_values[velocities].get_function_values(locally_relevant_solution,
                                                u_values);
      fe_values[velocities].get_function_gradients(locally_relevant_solution,
                                                   grad_u_values);

      fe_values[alpha].get_function_values(locally_relevant_solution,
                                           alpha_values);
      fe_values[alpha].get_function_gradients(locally_relevant_solution,
                                              grad_alpha_values);

      fe_values[alpha].get_function_values(locally_relevant_old_solution,
                                           alpha_old_values);

      for (unsigned int q = 0; q < n_q; ++q)
        {
          const double JxW = fe_values.JxW(q);

          const double div_u = trace(grad_u_values[q]);

          const double alpha_res =
            (alpha_values[q] - alpha_old_values[q]) / dt +
            (u_values[q] * grad_alpha_values[q]) + alpha_values[q] * div_u;


          const double solid_solid_phase_local_mass_source = alpha_res * JxW;

          local_solid_phase_continuity += solid_solid_phase_local_mass_source;

          local_solid_phase_max_local_continuity =
            std::max(local_solid_phase_max_local_continuity,
                     std::abs(solid_solid_phase_local_mass_source));

          local_mass_new += alpha_values[q] * JxW;
          local_mass_old += alpha_old_values[q] * JxW;
          local_volume += JxW;

          local_div_min = std::min(local_div_min, div_u);
          local_div_max = std::max(local_div_max, div_u);
          local_div_l2 += div_u * div_u * JxW;

          local_alpha_res_min = std::min(local_alpha_res_min, alpha_res);
          local_alpha_res_max = std::max(local_alpha_res_max, alpha_res);
          local_alpha_res_l2 += alpha_res * alpha_res * JxW;
        }

      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
           ++face)
        {
          if (!cell->face(face)->at_boundary())
            continue;

          fe_face_values.reinit(cell, face);

          fe_face_values[velocities].get_function_values(
            locally_relevant_solution, face_u_values);

          fe_face_values[alpha].get_function_values(locally_relevant_solution,
                                                    face_alpha_values);

          const types::boundary_id boundary_id =
            cell->face(face)->boundary_id();

          const auto boundary_it =
            parameters.boundary_conditions.type.find(boundary_id);

          AssertThrow(
            boundary_it != parameters.boundary_conditions.type.end(),
            ExcMessage(
              "No solid boundary condition specified for mesh boundary ID " +
              std::to_string(static_cast<unsigned int>(boundary_id))));

          const auto boundary_type = boundary_it->second;

          for (unsigned int q = 0; q < n_face_q; ++q)
            {
              const double JxW = fe_face_values.JxW(q);

              const Tensor<1, dim> normal = fe_face_values.normal_vector(q);

              const double un = face_u_values[q] * normal;

              const double flux = face_alpha_values[q] * un * JxW;

              // Function boundaries are currently treated as inlets.
              if (boundary_type == BoundaryConditions::BoundaryType::function)
                {
                  local_flux_inlet += flux;
                  local_flux_total += flux;
                }
              else if (boundary_type ==
                       BoundaryConditions::BoundaryType::outlet)
                {
                  local_flux_outlet += flux;
                  local_flux_total += flux;

                  if (un < 0.0)
                    {
                      local_outlet_backflow_flux += flux;
                      local_outlet_backflow_area += JxW;
                    }
                }
              else if (boundary_type ==
                         BoundaryConditions::BoundaryType::noslip ||
                       boundary_type == BoundaryConditions::BoundaryType::slip)
                {
                  local_flux_wall += flux;
                  local_flux_total += flux;
                }
              else if (boundary_type ==
                       BoundaryConditions::BoundaryType::periodic)
                {
                  // Periodic boundaries should not contribute to net flux.
                }
              else
                {
                  AssertThrow(
                    false,
                    ExcMessage(
                      "Unsupported solid boundary type in mass diagnostics for "
                      "boundary ID " +
                      std::to_string(static_cast<unsigned int>(boundary_id))));
                }
            }
        }
    }

  const double mass_new = Utilities::MPI::sum(local_mass_new, mpi_communicator);

  const double mass_old = Utilities::MPI::sum(local_mass_old, mpi_communicator);

  const double volume = Utilities::MPI::sum(local_volume, mpi_communicator);

  const double div_min = Utilities::MPI::min(local_div_min, mpi_communicator);

  const double div_max = Utilities::MPI::max(local_div_max, mpi_communicator);

  const double div_l2 =
    std::sqrt(Utilities::MPI::sum(local_div_l2, mpi_communicator));

  const double div_rms = div_l2 / std::sqrt(std::max(volume, 1e-30));

  const double alpha_res_min =
    Utilities::MPI::min(local_alpha_res_min, mpi_communicator);

  const double alpha_res_max =
    Utilities::MPI::max(local_alpha_res_max, mpi_communicator);

  const double alpha_res_l2 =
    std::sqrt(Utilities::MPI::sum(local_alpha_res_l2, mpi_communicator));

  const double alpha_res_rms =
    alpha_res_l2 / std::sqrt(std::max(volume, 1e-30));

  const double solid_phase_continuity =
    Utilities::MPI::sum(local_solid_phase_continuity, mpi_communicator);

  const double solid_phase_max_local_continuity =
    Utilities::MPI::max(local_solid_phase_max_local_continuity,
                        mpi_communicator);

  const double flux_total =
    Utilities::MPI::sum(local_flux_total, mpi_communicator);

  const double flux_wall =
    Utilities::MPI::sum(local_flux_wall, mpi_communicator);

  const double flux_inlet =
    Utilities::MPI::sum(local_flux_inlet, mpi_communicator);

  const double flux_outlet =
    Utilities::MPI::sum(local_flux_outlet, mpi_communicator);

  const double outlet_backflow_flux =
    Utilities::MPI::sum(local_outlet_backflow_flux, mpi_communicator);

  const double outlet_backflow_area =
    Utilities::MPI::sum(local_outlet_backflow_area, mpi_communicator);

  const double dMdt = (mass_new - mass_old) / dt;

  const double mass_balance_error = dMdt + flux_total;

  const double mass_balance_relative =
    std::abs(mass_balance_error) /
    (std::abs(dMdt) + std::abs(flux_total) + 1e-30);

  pcout << "solid diagnostics:\n"
        << "  alpha mass old      = " << mass_old << "\n"
        << "  alpha mass new      = " << mass_new << "\n"
        << "  dM/dt               = " << dMdt << "\n"
        << "  boundary flux total = " << flux_total << "\n"
        << "    wall flux         = " << flux_wall << "\n"
        << "    inlet flux        = " << flux_inlet << "\n"
        << "    outlet flux       = " << flux_outlet << "\n"
        << "  mass balance error  = " << mass_balance_error << "\n"
        << "  relative error      = " << mass_balance_relative << "\n"
        << "  solid_phase volume continuity error = " << solid_phase_continuity
        << "\n"
        << "  solid_phase max local continuity contribution = "
        << solid_phase_max_local_continuity << "\n"
        << "  div(u_s) min/max    = " << div_min << " , " << div_max << "\n"
        << "  div(u_s) L2/RMS     = " << div_l2 << " , " << div_rms << "\n"
        << "  alpha residual min/max = " << alpha_res_min << " , "
        << alpha_res_max << "\n"
        << "  alpha residual L2/RMS  = " << alpha_res_l2 << " , "
        << alpha_res_rms << "\n"
        << "  outlet backflow flux   = " << outlet_backflow_flux << "\n"
        << "  outlet backflow area   = " << outlet_backflow_area << std::endl;
}



template <int dim>
void
SolidPhaseSolver<dim>::assemble_local_system(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  SolidScratchData<dim>                                &scratch,
  SolidCopyData<dim>                                   &copy)
{
  copy.cell_is_local = cell->is_locally_owned();
  if (!copy.cell_is_local)
    return;

  copy.reset();

  AssertThrow(time_assembler != nullptr,
              ExcMessage("time_assembler is null in assemble_local_system."));
  AssertThrow(core_assembler != nullptr,
              ExcMessage("core_assembler is null in assemble_local_system."));

  AssertThrow(has_fluid_velocity_field,
              ExcMessage("Fluid velocity field must be set before assembly."));
  AssertThrow(picard_solution_ptr != nullptr,
              ExcMessage(
                "picard_solution_ptr is null in assemble_local_system."));
  AssertThrow(fluid_dof_handler_ptr != nullptr,
              ExcMessage("fluid_dof_handler_ptr is null."));
  AssertThrow(fluid_mapping_ptr != nullptr,
              ExcMessage("fluid_mapping_ptr is null."));
  AssertThrow(fluid_solution_ptr != nullptr,
              ExcMessage("fluid_solution_ptr is null."));

  const FEValuesExtractors::Vector solid_velocities(0);
  const FEValuesExtractors::Scalar solid_alpha(dim);
  const FEValuesExtractors::Vector fluid_velocities(0);
  const FEValuesExtractors::Scalar fluid_pressure(dim);

  typename DoFHandler<dim>::active_cell_iterator theta_cell(
    &(theta_dof_handler.get_triangulation()),
    cell->level(),
    cell->index(),
    &theta_dof_handler);

  typename DoFHandler<dim>::active_cell_iterator fluid_cell(
    &(fluid_dof_handler_ptr->get_triangulation()),
    cell->level(),
    cell->index(),
    fluid_dof_handler_ptr);

  scratch.reinit(cell,
                 theta_cell,
                 fluid_cell,
                 locally_relevant_old_solution,
                 locally_relevant_older_solution,
                 *picard_solution_ptr,
                 locally_relevant_fct_delta,
                 locally_relevant_theta_solution,
                 *fluid_solution_ptr,
                 solid_velocities,
                 solid_alpha,
                 fluid_velocities,
                 fluid_pressure);

  time_assembler->assemble(scratch, copy);
  core_assembler->assemble_cell(scratch, copy);



  for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
    {
      if (!cell->face(face)->at_boundary())
        continue;

      scratch.reinit_face(
        cell, face, *picard_solution_ptr, solid_velocities, solid_alpha);

      core_assembler->assemble_boundary_face(scratch,
                                             cell->face(face)->boundary_id(),
                                             copy);
    }


  // --------------------------------------------------
  // Lumped-mass RHS for the low-order alpha predictor:
  //   m_i * alpha_old_i / dt,  with  m_i = sum_j M_ij = integral(phi_i).
  // This is constraint-aware once distributed, so inlet/Dirichlet alpha
  // enters the predictor correctly.
  // --------------------------------------------------
  {
    const double dt_local = simulation_control->get_time_step();

    Vector<double> old_alpha_local(fe.n_dofs_per_cell());
    cell->get_dof_values(locally_relevant_old_solution, old_alpha_local);

    for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
      if (scratch.components[i] == dim) // alpha dofs only
        {
          double m_i = 0.0;
          for (unsigned int q = 0; q < scratch.n_q_points; ++q)
            m_i += scratch.phi_a[q][i] * scratch.JxW[q]; // integral(phi_i)

          copy.local_alpha_low_rhs(i) += m_i * old_alpha_local[i] / dt_local;

          // [REVIEW FIX 5] same lumped time RHS in the monolithic block
          copy.local_rhs(i) += m_i * old_alpha_local[i] / dt_local;
        }
  }

  // --------------------------------------------------
  // Build low-order alpha convection operator
  // using element-based artificial diffusion.
  // --------------------------------------------------

  const unsigned int n_dof = fe.n_dofs_per_cell();


  // First copy the original Galerkin convection matrix C.
  for (unsigned int i = 0; i < n_dof; ++i)
    {
      if (scratch.components[i] != dim)
        continue;

      for (unsigned int j = 0; j < n_dof; ++j)
        {
          if (scratch.components[j] != dim)
            continue;

          copy.local_alpha_low_matrix(i, j) +=
            copy.local_alpha_convection(i, j);
        }
    }


  // Add artificial diffusion.
  for (unsigned int i = 0; i < n_dof; ++i)
    {
      if (scratch.components[i] != dim)
        continue;

      for (unsigned int j = 0; j < n_dof; ++j)
        {
          if (scratch.components[j] != dim || i == j)
            continue;

          const double C_ij = copy.local_alpha_convection(i, j);

          const double C_ji = copy.local_alpha_convection(j, i);

          /*
           * Our convection operator is on the LHS.
           *
           * Equivalent to the paper's
           *
           * d_ij = max(-k_ij, 0, -k_ji)
           *
           * because C = -K in our sign convention.
           */
          const double d_ij = std::max({C_ij, 0.0, C_ji});

          copy.local_alpha_diffusion(i, j) += d_ij;

          // Off-diagonal diffusion
          copy.local_alpha_low_matrix(i, j) -= d_ij;

          // Preserve row sum / conservation
          copy.local_alpha_low_matrix(i, i) += d_ij;

          // [REVIEW FIX 5] the monolithic alpha-alpha block gets the same
          // upwinding, so it IS the low-order operator (lumped mass +
          // Galerkin convection + d_ij). At Picard convergence the alpha of
          // the monolithic solve equals alpha^L, and with the lagged
          // correction the momentum is in balance with the stored alpha^F.
          copy.local_matrix(i, j) -= d_ij;
          copy.local_matrix(i, i) += d_ij;
        }
    }

  cell->get_dof_indices(copy.local_dof_indices);
}


template <int dim>
void
SolidPhaseSolver<dim>::assemble_system()
{
  TimerOutput::Scope t(computing_timer, "assembly");

  AssertThrow(has_fluid_velocity_field,
              ExcMessage("Fluid velocity field must be set before assembly."));
  AssertThrow(picard_solution_ptr != nullptr,
              ExcMessage("picard_solution_ptr is null in assemble_system."));
  AssertThrow(fluid_dof_handler_ptr != nullptr,
              ExcMessage("fluid_dof_handler_ptr is null."));
  AssertThrow(fluid_mapping_ptr != nullptr,
              ExcMessage("fluid_mapping_ptr is null."));
  AssertThrow(fluid_solution_ptr != nullptr,
              ExcMessage("fluid_solution_ptr is null."));

  system_matrix = 0.0;
  system_rhs    = 0.0;

  alpha_low_matrix = 0.0;
  alpha_low_rhs    = 0.0;

  alpha_consistent_mass_matrix = 0.0;
  alpha_diffusion_matrix       = 0.0;

  locally_relevant_old_solution = old_solution;
  locally_relevant_old_solution.update_ghost_values();

  locally_relevant_older_solution = older_solution;
  locally_relevant_older_solution.update_ghost_values();



  const unsigned int n_q =
    std::max<unsigned int>(order + 1, order + parameters.quadrature_extra);

  const QGauss<dim>     cell_quadrature(n_q);
  const QGauss<dim - 1> face_quadrature(n_q);

  SolidScratchData<dim> scratch_data(fe,
                                     theta_fe,
                                     cell_quadrature,
                                     face_quadrature,
                                     *fluid_mapping_ptr,
                                     fluid_dof_handler_ptr->get_fe());

  AssertThrow(time_assembler != nullptr, ExcMessage("time_assembler is null."));
  AssertThrow(core_assembler != nullptr, ExcMessage("core_assembler is null."));

  const double time = simulation_control->get_current_time();

  if (parameters.source_term.enable)
    {
      parameters.source_term.function->set_time(time);
    }


  WorkStream::run(dof_handler.begin_active(),
                  dof_handler.end(),
                  *this,
                  &SolidPhaseSolver<dim>::assemble_local_system,
                  &SolidPhaseSolver<dim>::copy_local_system_to_global,
                  scratch_data,
                  SolidCopyData<dim>(fe.n_dofs_per_cell()));

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);

  alpha_low_matrix.compress(VectorOperation::add);
  alpha_low_rhs.compress(VectorOperation::add);

  alpha_consistent_mass_matrix.compress(VectorOperation::add);

  alpha_diffusion_matrix.compress(VectorOperation::add);


  // ------------------------------------------------------------
  // Build the BDF1 low-order RHS consistently with lumped mass:
  //
  //     M_L alpha^{n+1} / dt
  //       = M_L alpha^n / dt + ...
  //
  // m_i = sum_j M_ij
  // ------------------------------------------------------------
}


template <int dim>
void
SolidPhaseSolver<dim>::assemble_granular_temperature()
{
  TimerOutput::Scope t(computing_timer, "granular temperature assembly");

  static bool printed_theta_assembly_version = false;

  if (!printed_theta_assembly_version)
    {
      pcout << "Theta assembly: implicit collisional dissipation, "
            << "implicit viscous loss, explicit production" << std::endl;

      printed_theta_assembly_version = true;
    }

  AssertThrow(simulation_control->get_assembly_method() ==
                  Parameters::SimulationControl::TimeSteppingMethod::bdf1 ||
                simulation_control->get_assembly_method() ==
                  Parameters::SimulationControl::TimeSteppingMethod::steady_bdf,
              ExcMessage(
                "The granular-temperature implementation currently supports "
                "BDF1 only."));

  theta_matrix = 0.0;
  theta_rhs    = 0.0;



  const double dt      = simulation_control->get_time_step();
  const double c_theta = 1.5 * rho_s;

  // bool   use_theta_supg    = true;
  // double theta_supg_factor = 10.0;

  const double tiny_velocity      = 1e-12;
  const double tiny_cell_diameter = 1e-12;

  AssertThrow(dt > 0.0,
              ExcMessage(
                "The granular-temperature time step must be positive."));

  const unsigned int n_q =
    std::max<unsigned int>(order + 1, order + parameters.quadrature_extra);

  const QGauss<dim>     cell_quadrature(n_q);
  const QGauss<dim - 1> face_quadrature(n_q);

  FEValues<dim> solid_fe_values(
    fe, cell_quadrature, update_values | update_gradients | update_JxW_values);

  FEValues<dim> theta_fe_values(theta_fe,
                                cell_quadrature,
                                update_values | update_gradients |
                                  update_hessians | update_JxW_values);

  FEFaceValues<dim> solid_face_values(fe,
                                      face_quadrature,
                                      update_values | update_normal_vectors |
                                        update_JxW_values);

  FEFaceValues<dim> theta_face_values(theta_fe,
                                      face_quadrature,
                                      update_values | update_JxW_values);

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar alpha(dim);

  const unsigned int solid_n_q = cell_quadrature.size();

  const unsigned int face_n_q = face_quadrature.size();

  const unsigned int theta_dofs_per_cell = theta_fe.n_dofs_per_cell();

  std::vector<Tensor<1, dim>> picard_velocity_values(solid_n_q);

  std::vector<Tensor<2, dim>> picard_velocity_gradients(solid_n_q);

  std::vector<double> picard_alpha_values(solid_n_q);

  std::vector<Tensor<1, dim>> picard_alpha_gradients(solid_n_q);

  std::vector<double> old_alpha_values(solid_n_q);

  std::vector<double> picard_theta_values(solid_n_q);

  std::vector<Tensor<1, dim>> picard_theta_gradients(solid_n_q);

  std::vector<double> old_theta_values(solid_n_q);

  std::vector<Tensor<1, dim>> face_picard_velocity_values(face_n_q);

  std::vector<double> face_picard_alpha_values(face_n_q);

  FullMatrix<double> local_matrix(theta_dofs_per_cell, theta_dofs_per_cell);

  Vector<double> local_rhs(theta_dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(theta_dofs_per_cell);

  const double alpha_floor = 1e-6; // coefficient regularization only

  FullMatrix<double> local_transport(theta_dofs_per_cell, theta_dofs_per_cell);
  Vector<double>     lumped_capacity(theta_dofs_per_cell); // int c a/dt chi_i
  Vector<double>     lumped_sink(theta_dofs_per_cell);     // int S_sink chi_i
  Vector<double>     local_source(theta_dofs_per_cell);    // int P chi_i
  Vector<double>     local_theta_old(theta_dofs_per_cell); // nodal Theta^n


  // diagnostics
  double local_gamma_bracket_min = std::numeric_limits<double>::infinity();

  double local_gamma_bracket_max = -std::numeric_limits<double>::infinity();

  double local_C_gamma_min = std::numeric_limits<double>::infinity();

  double local_C_gamma_max = -std::numeric_limits<double>::infinity();

  double local_div_u_min = std::numeric_limits<double>::infinity();

  double local_div_u_max = -std::numeric_limits<double>::infinity();

  double local_theta_pic_min = std::numeric_limits<double>::infinity();

  double local_theta_pic_max = -std::numeric_limits<double>::infinity();

  double local_conductivity_min = std::numeric_limits<double>::infinity();

  double local_conductivity_max = -std::numeric_limits<double>::infinity();

  double local_C_viscous_min = std::numeric_limits<double>::infinity();

  double local_C_viscous_max = -std::numeric_limits<double>::infinity();

  double local_production_min = std::numeric_limits<double>::infinity();

  double local_production_max = -std::numeric_limits<double>::infinity();

  double local_frictional_pressure_min =
    std::numeric_limits<double>::infinity();

  double local_frictional_pressure_max =
    -std::numeric_limits<double>::infinity();

  double local_frictional_pressure_derivative_min =
    std::numeric_limits<double>::infinity();

  double local_frictional_pressure_derivative_max =
    -std::numeric_limits<double>::infinity();
  // diagnostics

  // for (const auto &solid_cell : dof_handler.active_cell_iterators())
  //   {
  //     if (!solid_cell->is_locally_owned())
  //       continue;


  //     typename DoFHandler<dim>::active_cell_iterator theta_cell(
  //       &(theta_dof_handler.get_triangulation()),
  //       solid_cell->level(),
  //       solid_cell->index(),
  //       &theta_dof_handler);

  //     solid_fe_values.reinit(solid_cell);
  //     theta_fe_values.reinit(theta_cell);



  //     solid_fe_values[velocities].get_function_values(
  //       locally_relevant_picard_solution, picard_velocity_values);

  //     solid_fe_values[velocities].get_function_gradients(
  //       locally_relevant_picard_solution, picard_velocity_gradients);

  //     solid_fe_values[alpha].get_function_values(
  //       locally_relevant_picard_solution, picard_alpha_values);

  //     solid_fe_values[alpha].get_function_gradients(
  //       locally_relevant_picard_solution, picard_alpha_gradients);


  //     solid_fe_values[alpha].get_function_values(locally_relevant_old_solution,
  //                                                old_alpha_values);


  //     theta_fe_values.get_function_values(
  //       locally_relevant_theta_picard_solution, picard_theta_values);

  //     theta_fe_values.get_function_gradients(
  //       locally_relevant_theta_picard_solution, picard_theta_gradients);

  //     theta_fe_values.get_function_values(locally_relevant_theta_old_solution,
  //                                         old_theta_values);

  //     local_matrix = 0.0;
  //     local_rhs    = 0.0;

  //     for (unsigned int q = 0; q < solid_n_q; ++q)
  //       {
  //         const double JxW = theta_fe_values.JxW(q);

  //         const double alpha_pic = picard_alpha_values[q];

  //         const double alpha_old = old_alpha_values[q];

  //         const double alpha_floor   = 1e-6;
  //         const double alpha_eff     = std::max(alpha_pic, alpha_floor);
  //         const double alpha_old_eff = std::max(alpha_old, alpha_floor);

  //         const double theta_pic = picard_theta_values[q];

  //         // const Tensor<1, dim> &grad_theta_pic =
  //         picard_theta_gradients[q];

  //         const double theta_old = old_theta_values[q];

  //         const Tensor<1, dim> &u_pic = picard_velocity_values[q];

  //         const Tensor<2, dim> &grad_u_pic = picard_velocity_gradients[q];

  //         const double div_u_pic = trace(grad_u_pic);

  //         const Tensor<1, dim> &grad_alpha_pic = picard_alpha_gradients[q];

  //         const double div_alpha_u_pic =
  //           (u_pic * grad_alpha_pic) + alpha_pic * div_u_pic;

  //         const double h = std::max(solid_cell->diameter(),
  //         tiny_cell_diameter);

  //         const double u_norm = std::max(u_pic.norm(), tiny_velocity);

  //         const double tau_theta =
  //           theta_supg_factor /
  //           std::sqrt((1.0 / dt) * (1.0 / dt) +
  //                     (2.0 * u_norm / h) * (2.0 * u_norm / h));



  //         const SolidKTGFValues ktgf =
  //           evaluate_solid_ktgf(parameters, alpha_pic, theta_pic, div_u_pic);


  //         const double gamma_bracket = (4.0 / parameters.particle_diameter) *
  //                                        std::sqrt(theta_pic / numbers::PI) -
  //                                      div_u_pic;

  //         const double C_gamma = ktgf.collisional_dissipation_coefficient;


  //         const double C_gamma_sink =
  //           std::max(C_gamma, 0.0);

  //         const double C_gamma_source =
  //           std::max(-C_gamma, 0.0);

  //         // const double C_gamma = 1;

  //         local_gamma_bracket_min =
  //           std::min(local_gamma_bracket_min, gamma_bracket);

  //         local_gamma_bracket_max =
  //           std::max(local_gamma_bracket_max, gamma_bracket);

  //         local_C_gamma_min = std::min(local_C_gamma_min, C_gamma);

  //         local_C_gamma_max = std::max(local_C_gamma_max, C_gamma);

  //         local_div_u_min = std::min(local_div_u_min, div_u_pic);

  //         local_div_u_max = std::max(local_div_u_max, div_u_pic);

  //         local_theta_pic_min = std::min(local_theta_pic_min, theta_pic);

  //         local_theta_pic_max = std::max(local_theta_pic_max, theta_pic);

  //         local_conductivity_min =
  //           std::min(local_conductivity_min, ktgf.conductivity);

  //         local_conductivity_max =
  //           std::max(local_conductivity_max, ktgf.conductivity);

  //         local_frictional_pressure_min =
  //           std::min(local_frictional_pressure_min,
  //           ktgf.frictional_pressure);

  //         local_frictional_pressure_max =
  //           std::max(local_frictional_pressure_max,
  //           ktgf.frictional_pressure);

  //         local_frictional_pressure_derivative_min =
  //           std::min(local_frictional_pressure_derivative_min,
  //                    ktgf.frictional_pressure_derivative);

  //         local_frictional_pressure_derivative_max =
  //           std::max(local_frictional_pressure_derivative_max,
  //                    ktgf.frictional_pressure_derivative);



  //         const Tensor<2, dim> tau_pic =
  //           make_solid_stress<dim>(ktgf.shear_viscosity,
  //                                  ktgf.bulk_viscosity,
  //                                  grad_u_pic);


  //         /*
  //         * ----------------------------------------------------------
  //         * Granular-temperature source/sink splitting
  //         * ----------------------------------------------------------
  //         *
  //         * Original production:
  //         *
  //         *   P_Theta
  //         *   =
  //         *   tau_s : grad(u_s)
  //         *   -
  //         *   p_s div(u_s)
  //         *
  //         * with
  //         *
  //         *   p_s = C_p Theta.
  //         *
  //         * Terms that remove granular energy are treated implicitly.
  //         * Terms that create granular energy are treated explicitly
  //         * using the Picard state.
  //         */


  //         /*
  //         * 1. Shear production
  //         *
  //         * tau_s : grad(u_s)
  //         *
  //         * For the KTGF stress used here this is the mechanical
  //         * production part.
  //         */
  //         const double shear_production = alpha_pic * scalar_product(tau_pic,
  //         grad_u_pic);


  //         /*
  //         * 2. Particle-pressure coefficient
  //         *
  //         * p_s =
  //         *
  //         * rho_s alpha Theta
  //         * [1 + 2 alpha g0 (1+e)]
  //         *
  //         *     = C_p Theta.
  //         *
  //         * Do not calculate this as particle_pressure/theta because
  //         * theta may be zero.
  //         */
  //         const double e_p =
  //           parameters.restitution_coefficient;

  //         const double C_particle_pressure =
  //           rho_s *
  //           alpha_pic *
  //           (1.0 +
  //           2.0 *
  //             alpha_pic *
  //             ktgf.radial_distribution *
  //             (1.0 + e_p));


  //         /*
  //         * Pressure-work contribution:
  //         *
  //         *   -p_s div(u_s)
  //         *
  //         * =
  //         *
  //         *   -C_p div(u_s) Theta.
  //         *
  //         * Define
  //         *
  //         *   C_pressure = C_p div(u_s).
  //         *
  //         * If C_pressure > 0:
  //         *     pressure work is a sink.
  //         *
  //         * If C_pressure < 0:
  //         *     pressure work is a source.
  //         */
  //         const double C_pressure =
  //           C_particle_pressure *
  //           div_u_pic;

  //         const double C_pressure_sink =
  //           std::max(C_pressure, 0.0);

  //         const double C_pressure_source =
  //           std::max(-C_pressure, 0.0);


  //         /*
  //         * 3. Collisional dissipation
  //         *
  //         * gamma_s = C_gamma Theta.
  //         *
  //         * C_gamma can have either sign because of
  //         *
  //         *   4/d_p sqrt(Theta/pi) - div(u_s).
  //         */
  //         // const double C_gamma =
  //         //   ktgf.collisional_dissipation_coefficient;

  //         // const double C_gamma_sink =
  //         //   std::max(C_gamma, 0.0);

  //         // const double C_gamma_source =
  //         //   std::max(-C_gamma, 0.0);


  //         /*
  //         * 4. Fluid viscous damping
  //         *
  //         * -J_vis =
  //         *
  //         * 3 alpha beta Theta.
  //         *
  //         * beta >= 0 and alpha >= 0, therefore this is always
  //         * a sink.
  //         */
  //         const double C_viscous =
  //           3.0 *
  //           alpha_pic *
  //           beta;


  //         /*
  //         * Total coefficient treated implicitly:
  //         *
  //         *   C_sink Theta^{k+1}.
  //         */
  //         const double C_theta_sink =
  //           C_viscous
  //           + C_gamma_sink
  //           + C_pressure_sink;


  //         /*
  //         * Explicit positive source evaluated from the Picard state:
  //         *
  //         *   S_theta^k =
  //         *
  //         *   tau^k : grad(u^k)
  //         *   +
  //         *   C_gamma_source Theta^k
  //         *   +
  //         *   C_pressure_source Theta^k.
  //         */
  //         const double theta_source =
  //           shear_production
  //           +
  //           (C_gamma_source +
  //           C_pressure_source) *
  //             theta_pic;


  //         /*
  //         * Original total production at the Picard state.
  //         *
  //         * Keep this only for diagnostics.
  //         */
  //         const double production =
  //           shear_production
  //           -
  //           ktgf.particle_pressure *
  //           div_u_pic;


  //         // const double C_viscous = 0;

  //         local_production_min = std::min(local_production_min, production);

  //         local_production_max = std::max(local_production_max, production);

  //         local_C_viscous_min = std::min(local_C_viscous_min, C_viscous);
  //         local_C_viscous_max = std::max(local_C_viscous_max, C_viscous);



  //         AssertThrow(std::isfinite(ktgf.conductivity),
  //                     ExcMessage("Non-finite granular-temperature "
  //                                "conductivity."));

  //         AssertThrow(std::isfinite(production),
  //                     ExcMessage("Non-finite granular-energy "
  //                                "production term."));

  //         AssertThrow(std::isfinite(C_gamma),
  //                     ExcMessage("Non-finite collisional-dissipation "
  //                                "coefficient."));

  //         AssertThrow(std::isfinite(C_viscous),
  //                     ExcMessage("Non-finite viscous-loss "
  //                                "coefficient."));

  //         // AssertThrow(std::isfinite(damping_coefficient),
  //         //             ExcMessage("Non-finite granular-temperature "
  //         //                        "damping coefficient."));

  //         for (unsigned int i = 0; i < theta_dofs_per_cell; ++i)
  //           {
  //             const double chi_i = theta_fe_values.shape_value(i, q);

  //             const Tensor<1, dim> grad_chi_i =
  //               theta_fe_values.shape_grad(i, q);

  //             const double supg_test_theta = u_pic * grad_chi_i;

  //             for (unsigned int j = 0; j < theta_dofs_per_cell; ++j)
  //               {
  //                 const double chi_j = theta_fe_values.shape_value(j, q);

  //                 const Tensor<1, dim> grad_chi_j =
  //                   theta_fe_values.shape_grad(j, q);

  //                 const Tensor<2, dim> hessian_chi_j =
  //                   theta_fe_values.shape_hessian(j, q);

  //                 const double laplacian_chi_j = trace(hessian_chi_j);


  //                 local_matrix(i, j) +=
  //                   (c_theta * alpha_eff / dt) * chi_i * chi_j * JxW;


  //                 local_matrix(i, j) +=
  //                   -c_theta * alpha_eff * chi_j * (grad_chi_i * u_pic) *
  //                   JxW;



  //                 local_matrix(i, j) +=
  //                   ktgf.conductivity * (grad_chi_i * grad_chi_j) * JxW;

  //                 // local_matrix(i, j) += C_gamma * chi_i * chi_j * JxW;



  //                 // local_matrix(i, j) += C_viscous * chi_i * chi_j * JxW;

  //                 /*
  //                 * Implicit non-negative granular-energy sinks.
  //                 */
  //                 local_matrix(i, j) +=
  //                   C_theta_sink *
  //                   chi_i *
  //                   chi_j *
  //                   JxW;


  //                 if (use_theta_supg)
  //                   {
  //                     const double strong_theta_trial =
  //                       c_theta *
  //                         (alpha_eff * chi_j / dt +
  //                         alpha_eff * (u_pic * grad_chi_j) +
  //                         div_alpha_u_pic * chi_j)
  //                       -
  //                       ktgf.conductivity * laplacian_chi_j
  //                       +
  //                       C_theta_sink * chi_j;

  //                     local_matrix(i, j) +=
  //                       tau_theta * supg_test_theta * strong_theta_trial *
  //                       JxW;
  //                   }
  //               }


  //             local_rhs(i) +=
  //               (c_theta / dt) * chi_i * alpha_old_eff * theta_old * JxW;


  //             /*
  //             * Explicit granular-energy production/source.
  //             */
  //             local_rhs(i) +=
  //               chi_i *
  //               theta_source *
  //               JxW;

  //             // local_rhs(i) += -C_gamma * chi_i * theta_pic * JxW;

  //             if (use_theta_supg)
  //               {
  //                 // const double strong_theta_rhs =
  //                 //   (c_theta / dt) * alpha_old * theta_old + production;

  //                 const double strong_theta_rhs =
  //                   (c_theta / dt) *
  //                     alpha_old_eff *
  //                     theta_old
  //                   +
  //                   theta_source;

  //                 local_rhs(i) +=
  //                   tau_theta * supg_test_theta * strong_theta_rhs * JxW;
  //               }
  //           }
  //       }


  //     for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
  //          ++face)
  //       {
  //         if (!solid_cell->face(face)->at_boundary())
  //           continue;

  //         const types::boundary_id boundary_id =
  //           solid_cell->face(face)->boundary_id();

  //         const auto boundary_it =
  //           parameters.boundary_conditions.type.find(boundary_id);

  //         AssertThrow(boundary_it !=
  //         parameters.boundary_conditions.type.end(),
  //                     ExcMessage("No solid boundary condition specified "
  //                                "for boundary ID " +
  //                                std::to_string(
  //                                  static_cast<unsigned int>(boundary_id))));

  //         if (boundary_it->second !=
  //         BoundaryConditions::BoundaryType::outlet)
  //           continue;

  //         solid_face_values.reinit(solid_cell, face);

  //         theta_face_values.reinit(theta_cell, face);

  //         solid_face_values[velocities].get_function_values(
  //           locally_relevant_picard_solution, face_picard_velocity_values);

  //         solid_face_values[alpha].get_function_values(
  //           locally_relevant_picard_solution, face_picard_alpha_values);

  //         for (unsigned int q = 0; q < face_n_q; ++q)
  //           {
  //             const Tensor<1, dim> normal =
  //             solid_face_values.normal_vector(q);

  //             const double normal_velocity =
  //               face_picard_velocity_values[q] * normal;

  //             if (normal_velocity <= 0.0)
  //               continue;

  //             const double JxW = theta_face_values.JxW(q);

  //             for (unsigned int i = 0; i < theta_dofs_per_cell; ++i)
  //               {
  //                 const double chi_i = theta_face_values.shape_value(i, q);

  //                 for (unsigned int j = 0; j < theta_dofs_per_cell; ++j)
  //                   {
  //                     const double chi_j = theta_face_values.shape_value(j,
  //                     q);

  //                     local_matrix(i, j) +=
  //                       c_theta * face_picard_alpha_values[q] * chi_i * chi_j
  //                       * normal_velocity * JxW;
  //                   }
  //               }
  //           }
  //       }

  //     theta_cell->get_dof_indices(local_dof_indices);

  //     theta_constraints.distribute_local_to_global(
  //       local_matrix, local_rhs, local_dof_indices, theta_matrix, theta_rhs);
  //   }

  for (const auto &solid_cell : dof_handler.active_cell_iterators())
    {
      if (!solid_cell->is_locally_owned())
        continue;

      typename DoFHandler<dim>::active_cell_iterator theta_cell(
        &(theta_dof_handler.get_triangulation()),
        solid_cell->level(),
        solid_cell->index(),
        &theta_dof_handler);

      solid_fe_values.reinit(solid_cell);
      theta_fe_values.reinit(theta_cell);

      solid_fe_values[velocities].get_function_values(
        locally_relevant_picard_solution, picard_velocity_values);

      solid_fe_values[velocities].get_function_gradients(
        locally_relevant_picard_solution, picard_velocity_gradients);

      solid_fe_values[alpha].get_function_values(
        locally_relevant_picard_solution, picard_alpha_values);

      theta_fe_values.get_function_values(
        locally_relevant_theta_picard_solution, picard_theta_values);

      // Nodal Theta^n: the lumped time term uses nodal values, not
      // quadrature values, so the RHS is exactly m_i * Theta^n_i >= 0.
      theta_cell->get_dof_values(locally_relevant_theta_old_solution,
                                 local_theta_old);

      local_matrix    = 0.0;
      local_rhs       = 0.0;
      local_transport = 0.0;
      lumped_capacity = 0.0;
      lumped_sink     = 0.0;
      local_source    = 0.0;

      for (unsigned int q = 0; q < solid_n_q; ++q)
        {
          const double JxW = theta_fe_values.JxW(q);

          const double alpha_pic = picard_alpha_values[q];
          const double alpha_eff = std::max(alpha_pic, alpha_floor);
          const double theta_pic =
            picard_theta_values[q]; // >= 0 (Q1, nodal >= 0)

          const Tensor<1, dim> &u_pic      = picard_velocity_values[q];
          const Tensor<2, dim> &grad_u_pic = picard_velocity_gradients[q];
          const double          div_u_pic  = trace(grad_u_pic);

          const SolidKTGFValues ktgf =
            evaluate_solid_ktgf(parameters, alpha_pic, theta_pic, div_u_pic);

          // ---- Production, as a sum of squares (>= 0 by construction) -----
          // For tau = mu (grad u + grad u^T) + (lambda - 2/3 mu) div(u) I
          // (make_solid_stress), with S_dev = sym(grad u) - div(u)/dim I:
          //   tau : grad u = 2 mu S_dev:S_dev
          //                + (lambda + 2 mu (1/dim - 1/3)) div(u)^2
          // which is identical to scalar_product(tau, grad u), and manifestly
          // non-negative because mu, lambda >= 0 and 1/dim >= 1/3.
          Tensor<2, dim> S_dev = 0.5 * (grad_u_pic + transpose(grad_u_pic));
          for (unsigned int d = 0; d < dim; ++d)
            S_dev[d][d] -= div_u_pic / dim;

          const double bulk_coefficient =
            ktgf.bulk_viscosity +
            2.0 * ktgf.shear_viscosity * (1.0 / dim - 1.0 / 3.0);

          const double shear_production =
            alpha_pic *
            (2.0 * ktgf.shear_viscosity * scalar_product(S_dev, S_dev) +
             bulk_coefficient * div_u_pic * div_u_pic);

          // ---- Pressure work  -p_s div(u) = -C_p div(u) Theta -------------
          const double e_p = parameters.restitution_coefficient;

          // [REVIEW FIX 3] collisional dissipation (Passalacqua & Fox A.7):
          //   gamma = 3(1-e^2) rho alpha^2 g0 Theta [(4/d) sqrt(Theta/pi)
          //                                          - div(u)]
          // The -div(u) part was missing. It has the same form as the
          // pressure work, so the two are combined:
          //   -p_s div u - gamma_div = -C_eff Theta div u,
          //   C_eff = rho alpha [1 + 2(1+e) alpha g0 - 3(1-e^2) alpha g0]
          //         = rho alpha [1 + (1+e)(3e-1) alpha g0].
          // Same sign split as before (sink if div u > 0, source if < 0).
          const double C_particle_pressure =
            rho_s * alpha_pic *
            (1.0 + (1.0 + e_p) * (3.0 * e_p - 1.0) * alpha_pic *
                     ktgf.radial_distribution);

          const double C_pressure      = C_particle_pressure * div_u_pic;
          const double C_pressure_sink = std::max(C_pressure, 0.0); // implicit
          const double C_pressure_source =
            std::max(-C_pressure, 0.0); // explicit

          // ---- Collisional dissipation (pure-sink form) and viscous loss ---
          const double C_gamma      = ktgf.collisional_dissipation_coefficient;
          const double C_gamma_sink = std::max(C_gamma, 0.0);
          const double C_gamma_source =
            std::max(-C_gamma, 0.0); // 0 for sink form

          const double C_viscous = 3.0 * alpha_pic * beta; // >= 0

          // ---- Split: implicit sinks (>= 0) and explicit sources (>= 0) ----
          const double S_sink = C_viscous + C_gamma_sink + C_pressure_sink;

          const double P_source =
            shear_production + (C_gamma_source + C_pressure_source) * theta_pic;

          const double capacity = c_theta * alpha_eff / dt; // > 0

          AssertThrow(std::isfinite(ktgf.conductivity) &&
                        std::isfinite(S_sink) && std::isfinite(P_source),
                      ExcMessage(
                        "Non-finite granular-temperature coefficient."));

          // ---- Diagnostics (same quantities as before) ----------------------
          const double gamma_bracket = (4.0 / parameters.particle_diameter) *
                                         std::sqrt(theta_pic / numbers::PI) -
                                       div_u_pic;

          const double production =
            shear_production - ktgf.particle_pressure * div_u_pic;

          // [REVIEW FIX 10] diagnostics only where particles are present;
          // values in empty cells (alpha ~ 0) have no physical meaning and
          // were the ones reported as max|div u|, max Theta, ...
          if (alpha_pic > 1e-3)
            {
              local_gamma_bracket_min =
                std::min(local_gamma_bracket_min, gamma_bracket);
              local_gamma_bracket_max =
                std::max(local_gamma_bracket_max, gamma_bracket);
              local_C_gamma_min   = std::min(local_C_gamma_min, C_gamma);
              local_C_gamma_max   = std::max(local_C_gamma_max, C_gamma);
              local_div_u_min     = std::min(local_div_u_min, div_u_pic);
              local_div_u_max     = std::max(local_div_u_max, div_u_pic);
              local_theta_pic_min = std::min(local_theta_pic_min, theta_pic);
              local_theta_pic_max = std::max(local_theta_pic_max, theta_pic);
              local_conductivity_min =
                std::min(local_conductivity_min, ktgf.conductivity);
              local_conductivity_max =
                std::max(local_conductivity_max, ktgf.conductivity);
              local_C_viscous_min  = std::min(local_C_viscous_min, C_viscous);
              local_C_viscous_max  = std::max(local_C_viscous_max, C_viscous);
              local_production_min = std::min(local_production_min, production);
              local_production_max = std::max(local_production_max, production);
              local_frictional_pressure_min =
                std::min(local_frictional_pressure_min,
                         ktgf.frictional_pressure);
              local_frictional_pressure_max =
                std::max(local_frictional_pressure_max,
                         ktgf.frictional_pressure);
              local_frictional_pressure_derivative_min =
                std::min(local_frictional_pressure_derivative_min,
                         ktgf.frictional_pressure_derivative);
              local_frictional_pressure_derivative_max =
                std::max(local_frictional_pressure_derivative_max,
                         ktgf.frictional_pressure_derivative);
            }

          // ---- Element contributions ---------------------------------------
          for (unsigned int i = 0; i < theta_dofs_per_cell; ++i)
            {
              const double         chi_i = theta_fe_values.shape_value(i, q);
              const Tensor<1, dim> grad_chi_i =
                theta_fe_values.shape_grad(i, q);

              // Row-sum lumping: sum_j int w chi_i chi_j = int w chi_i.
              lumped_capacity(i) += capacity * chi_i * JxW;
              lumped_sink(i) += S_sink * chi_i * JxW;
              local_source(i) += P_source * chi_i * JxW;

              for (unsigned int j = 0; j < theta_dofs_per_cell; ++j)
                {
                  const Tensor<1, dim> grad_chi_j =
                    theta_fe_values.shape_grad(j, q);

                  // Advective convection c a u.grad(Theta): sum_j = 0.
                  local_transport(i, j) +=
                    c_theta * alpha_eff * chi_i * (u_pic * grad_chi_j) * JxW;

                  // Conduction: sum_j = 0.
                  local_transport(i, j) +=
                    ktgf.conductivity * (grad_chi_i * grad_chi_j) * JxW;
                }
            }
        }

      // ---- Discrete upwinding: every off-diagonal <= 0, row sums kept 0 ----
      for (unsigned int i = 0; i < theta_dofs_per_cell; ++i)
        for (unsigned int j = i + 1; j < theta_dofs_per_cell; ++j)
          {
            const double d_ij =
              std::max({0.0, local_transport(i, j), local_transport(j, i)});

            local_transport(i, j) -= d_ij;
            local_transport(j, i) -= d_ij;
            local_transport(i, i) += d_ij;
            local_transport(j, j) += d_ij;
          }

      // ---- Assemble: lumped diagonal + transport; non-negative RHS ---------
      for (unsigned int i = 0; i < theta_dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < theta_dofs_per_cell; ++j)
            local_matrix(i, j) = local_transport(i, j);

          local_matrix(i, i) += lumped_capacity(i) + lumped_sink(i);

          local_rhs(i) =
            lumped_capacity(i) * local_theta_old(i) + local_source(i);
        }

      // No outlet face integral: the advective form needs no convective
      // boundary term (outflow is natural; walls have u.n = 0).

      theta_cell->get_dof_indices(local_dof_indices);

      theta_constraints.distribute_local_to_global(
        local_matrix, local_rhs, local_dof_indices, theta_matrix, theta_rhs);
    }


  const double global_gamma_bracket_min =
    Utilities::MPI::min(local_gamma_bracket_min, mpi_communicator);

  const double global_gamma_bracket_max =
    Utilities::MPI::max(local_gamma_bracket_max, mpi_communicator);

  const double global_C_gamma_min =
    Utilities::MPI::min(local_C_gamma_min, mpi_communicator);

  const double global_C_gamma_max =
    Utilities::MPI::max(local_C_gamma_max, mpi_communicator);

  const double global_div_u_min =
    Utilities::MPI::min(local_div_u_min, mpi_communicator);

  const double global_div_u_max =
    Utilities::MPI::max(local_div_u_max, mpi_communicator);

  const double global_theta_pic_min =
    Utilities::MPI::min(local_theta_pic_min, mpi_communicator);

  const double global_theta_pic_max =
    Utilities::MPI::max(local_theta_pic_max, mpi_communicator);

  const double conductivity_min =
    Utilities::MPI::min(local_conductivity_min, mpi_communicator);
  const double conductivity_max =
    Utilities::MPI::max(local_conductivity_max, mpi_communicator);
  const double C_viscous_min =
    Utilities::MPI::min(local_C_viscous_min, mpi_communicator);
  const double C_viscous_max =
    Utilities::MPI::max(local_C_viscous_max, mpi_communicator);
  const double production_min =
    Utilities::MPI::min(local_production_min, mpi_communicator);
  const double production_max =
    Utilities::MPI::max(local_production_max, mpi_communicator);

  if (parameters.solver_verbose)
    {
      pcout << "  theta_pic min/max      = " << global_theta_pic_min << " / "
            << global_theta_pic_max << '\n'
            << "  div(u_s) min/max       = " << global_div_u_min << " / "
            << global_div_u_max << '\n'
            << "  gamma bracket min/max  = " << global_gamma_bracket_min
            << " / " << global_gamma_bracket_max << '\n'
            << "  C_gamma min/max        = " << global_C_gamma_min << " / "
            << global_C_gamma_max << '\n'
            << "  conductivity min/max   = " << conductivity_min << " / "
            << conductivity_max << '\n'
            << "  C_viscous min/max      = " << C_viscous_min << " / "
            << C_viscous_max << '\n'
            << "  production min/max     = " << production_min << " / "
            << production_max << std::endl;
    }

  theta_matrix.compress(VectorOperation::add);

  theta_rhs.compress(VectorOperation::add);
}

template <int dim>
void
SolidPhaseSolver<dim>::solve_granular_temperature()
{
  TimerOutput::Scope t(computing_timer, "theta solve");

  if (parameters.solver_verbose)
    pcout << "Solving granular-temperature system... " << std::flush;

  TrilinosWrappers::MPI::Vector distributed_theta(theta_locally_owned,
                                                  mpi_communicator);
  distributed_theta = theta_solution;

  theta_constraints.distribute(distributed_theta);
  distributed_theta.compress(VectorOperation::insert);

  // theta_ilu = std::make_shared<TrilinosWrappers::PreconditionILU>();
  // TrilinosWrappers::PreconditionILU::AdditionalData ilu_data;
  // ilu_data.overlap = parameters.ilu_overlap;
  // theta_ilu->initialize(theta_matrix, ilu_data);

  // PrimitiveVectorMemory<TrilinosWrappers::MPI::Vector> memory;

  // const double rhs_norm = theta_rhs.l2_norm();
  // const double tolerance =
  //   std::max(parameters.solver_abs_tol, parameters.solver_rel_tol *
  //   rhs_norm);

  // SolverControl solver_control(parameters.solver_max_it, tolerance);

  // SolverFGMRES<TrilinosWrappers::MPI::Vector> solver(
  //   solver_control,
  //   memory,
  //   SolverFGMRES<TrilinosWrappers::MPI::Vector>::AdditionalData(
  //     parameters.solver_restart));

  // solver.solve(theta_matrix, distributed_theta, theta_rhs, *theta_ilu);

  TrilinosWrappers::MPI::Vector completely_distributed_theta(
    theta_locally_owned, mpi_communicator);

  SolverControl                                  direct_control(1, 0.0);
  TrilinosWrappers::SolverDirect::AdditionalData direct_data(false,
                                                             "Amesos_Klu");
  TrilinosWrappers::SolverDirect direct_solver(direct_control, direct_data);

  direct_solver.solve(theta_matrix, completely_distributed_theta, theta_rhs);

  theta_constraints.distribute(completely_distributed_theta);


  // theta_constraints.distribute(distributed_theta);
  // distributed_theta.compress(VectorOperation::insert);

  theta_solution = completely_distributed_theta;
  theta_solution.compress(VectorOperation::insert);

  locally_relevant_theta_solution = theta_solution;
  locally_relevant_theta_solution.update_ghost_values();

  if (parameters.solver_verbose)
    {
      // Direct solve: report the relative residual as a sanity check.
      // It should be at round-off level (~1e-14).
      TrilinosWrappers::MPI::Vector residual(theta_locally_owned,
                                             mpi_communicator);
      theta_matrix.vmult(residual, completely_distributed_theta);
      residual -= theta_rhs;

      pcout << "direct solve (Amesos_Klu), relative residual = "
            << residual.l2_norm() / std::max(theta_rhs.l2_norm(), 1e-300)
            << std::endl;
    }
}


template <int dim>
void
SolidPhaseSolver<dim>::setup_time_assembler()
{
  time_assembler =
    std::make_shared<SolidAssemblerBDF<dim>>(simulation_control, parameters);
}

template <int dim>
void
SolidPhaseSolver<dim>::setup_core_assembler()
{
  core_assembler =
    std::make_shared<SolidCoreAssembler<dim>>(parameters, simulation_control);
}

template <int dim>
void
SolidPhaseSolver<dim>::copy_local_system_to_global(
  const SolidCopyData<dim> &copy)
{
  if (!copy.cell_is_local)
    return;

  constraints.distribute_local_to_global(copy.local_matrix,
                                         copy.local_rhs,
                                         copy.local_dof_indices,
                                         system_matrix,
                                         system_rhs);

  constraints.distribute_local_to_global(copy.local_alpha_low_matrix,
                                         copy.local_alpha_low_rhs,
                                         copy.local_dof_indices,
                                         alpha_low_matrix,
                                         alpha_low_rhs);


  constraints.distribute_local_to_global(copy.local_alpha_consistent_mass,
                                         copy.local_dof_indices,
                                         alpha_consistent_mass_matrix);

  constraints.distribute_local_to_global(copy.local_alpha_diffusion,
                                         copy.local_dof_indices,
                                         alpha_diffusion_matrix);
}



template <int dim>
void
SolidPhaseSolver<dim>::solve()
{
  TimerOutput::Scope t(computing_timer, "solve");


  if (parameters.solver_verbose)
    pcout << "Solving solid system... " << std::flush;

  TrilinosWrappers::MPI::BlockVector distributed_solution(owned_partitioning,
                                                          mpi_communicator);
  distributed_solution = solution;

  constraints.distribute(distributed_solution);
  distributed_solution.compress(VectorOperation::insert);

  PrimitiveVectorMemory<TrilinosWrappers::MPI::BlockVector> mem;

  const double rhs_norm = system_rhs.l2_norm();
  const double tol =
    std::max(parameters.solver_abs_tol, parameters.solver_rel_tol * rhs_norm);

  SolverControl solver_control(parameters.solver_max_it, tol);



  setup_linear_preconditioners();



  const bool use_amg = (parameters.amg_sweeps > 0);

  SolidBlockDiagonalPreconditioner preconditioner(velocity_amg,
                                                  velocity_ilu,
                                                  alpha_ilu,
                                                  use_amg);

  SolverFGMRES<TrilinosWrappers::MPI::BlockVector> solver(
    solver_control,
    mem,
    SolverFGMRES<TrilinosWrappers::MPI::BlockVector>::AdditionalData(
      parameters.solver_restart));

  try
    {
      solver.solve(system_matrix,
                   distributed_solution,
                   system_rhs,
                   preconditioner);
    }
  catch (SolverControl::NoConvergence &)
    {
      if (parameters.solver_verbose)
        pcout << "\nNoConvergence: retrying with relaxed settings...\n";

      SolverControl retry_control(parameters.solver_max_it, tol);

      SolverFGMRES<TrilinosWrappers::MPI::BlockVector> retry_solver(
        retry_control,
        mem,
        SolverFGMRES<TrilinosWrappers::MPI::BlockVector>::AdditionalData(
          std::max(parameters.solver_restart, 200u)));

      retry_solver.solve(system_matrix,
                         distributed_solution,
                         system_rhs,
                         preconditioner);

      solver_control = retry_control;
    }

  // [REVIEW FIX 8] true residual. FGMRES reports an estimate, which is
  // exactly 0 after a breakdown even when x is not a solution.
  TrilinosWrappers::MPI::BlockVector true_r(owned_partitioning,
                                            mpi_communicator);
  system_matrix.vmult(true_r, distributed_solution);
  true_r.sadd(-1.0, 1.0, system_rhs); // r = b - A x
  const double true_residual = true_r.l2_norm();

  constraints.distribute(distributed_solution);
  distributed_solution.compress(VectorOperation::insert);

  solution = distributed_solution;
  solution.compress(VectorOperation::insert);

  locally_relevant_solution = solution;
  locally_relevant_solution.update_ghost_values();

  if (parameters.solver_verbose)
    pcout << solver_control.last_step()
          << " iterations. residual=" << solver_control.last_value()
          << " true residual=" << true_residual << std::endl;


  const double true_tol = std::max(10.0 * tol, 1e-6 * rhs_norm);

  if (true_residual > true_tol)
    {
      std::ostringstream msg;
      msg << std::scientific << "Solid linear solve failed: true residual "
          << true_residual << " > " << true_tol << " (||b|| = " << rhs_norm
          << ")";
      AssertThrow(false, ExcMessage(msg.str()));
    }
}

template <int dim>
void
SolidPhaseSolver<dim>::solve_alpha_low_order()
{
  AssertThrow(simulation_control->get_assembly_method() ==
                  Parameters::SimulationControl::TimeSteppingMethod::bdf1 ||
                simulation_control->get_assembly_method() ==
                  Parameters::SimulationControl::TimeSteppingMethod::steady_bdf,
              ExcMessage(
                "Low-order alpha predictor currently supports BDF1 only."));


  const auto &A = alpha_low_matrix.block(1, 1);

  const auto &b = alpha_low_rhs.block(1);


  /*
   * Start Jacobi from the previous admissible Picard iterate,
   * not from the high-order Galerkin solution.
   */
  alpha_low_solution = picard_solution;
  alpha_low_solution.compress(VectorOperation::insert);


  /*
   * Verify the algebraic conditions required by the
   * positivity-preserving low-order scheme:
   *
   *   A_ii > 0
   *   A_ij <= 0, i != j
   *   b_i >= 0
   *
   * These checks should remain during development.
   */
  double local_min_diagonal = std::numeric_limits<double>::max();

  double local_max_offdiag = -std::numeric_limits<double>::max();

  double local_min_rhs = std::numeric_limits<double>::max();


  for (const auto i : alpha_low_solution.block(1).locally_owned_elements())
    {
      const double a_ii = A.diag_element(i);

      local_min_diagonal = std::min(local_min_diagonal, a_ii);

      local_min_rhs = std::min(local_min_rhs, static_cast<double>(b[i]));

      for (auto entry = A.begin(i); entry != A.end(i); ++entry)
        {
          const auto j = entry->column();

          if (j == i)
            continue;

          local_max_offdiag = std::max(local_max_offdiag, entry->value());
        }
    }


  const double global_min_diagonal =
    Utilities::MPI::min(local_min_diagonal, mpi_communicator);

  const double global_max_offdiag =
    Utilities::MPI::max(local_max_offdiag, mpi_communicator);

  const double global_min_rhs =
    Utilities::MPI::min(local_min_rhs, mpi_communicator);


  AssertThrow(global_min_diagonal > 0.0,
              ExcMessage(
                "Low-order alpha matrix has a non-positive diagonal."));

  AssertThrow(global_max_offdiag <= 1e-14,
              ExcMessage("Low-order alpha matrix does not have "
                         "non-positive off-diagonal entries."));

  AssertThrow(global_min_rhs >= -1e-14,
              ExcMessage("Low-order alpha RHS is negative."));


  const double rhs_norm = b.l2_norm();

  const double tolerance =
    std::max(parameters.solver_abs_tol, parameters.solver_rel_tol * rhs_norm);

  SolverControl solver_control(parameters.solver_max_it, tolerance);


  /*
   * Point Jacobi preconditioner.
   *
   * One sweep and omega = 1 gives D^{-1}r.
   */
  TrilinosWrappers::PreconditionJacobi jacobi;

  TrilinosWrappers::PreconditionJacobi::AdditionalData jacobi_data(
    1.0, // Jacobi relaxation
    0.0, // do not modify diagonal
    1);  // one sweep

  jacobi.initialize(A, jacobi_data);


  /*
   * Richardson:
   *
   * x^(m+1) = x^m + D^{-1}(b-Ax^m)
   *
   * which is standard Jacobi for omega = 1.
   */
  SolverRichardson<TrilinosWrappers::MPI::Vector> solver(
    solver_control,
    SolverRichardson<TrilinosWrappers::MPI::Vector>::AdditionalData(1.0));


  solver.solve(A, alpha_low_solution.block(1), b, jacobi);


  alpha_low_solution.block(1).compress(VectorOperation::insert);

  constraints.distribute(alpha_low_solution);
  alpha_low_solution.compress(VectorOperation::insert);


  if (parameters.solver_verbose)
    pcout << "Solving low-order alpha system... " << solver_control.last_step()
          << " iterations. residual=" << solver_control.last_value()
          << std::endl;
}

template <int dim>
void
SolidPhaseSolver<dim>::apply_alpha_fct()
{
  AssertThrow(order == 1,
              ExcMessage("Current alpha FCT implementation supports Q1 only."));

  AssertThrow(!parameters.source_term.enable,
              ExcMessage(
                "Current alpha FCT implementation assumes zero alpha source."));

  const double dt = simulation_control->get_time_step();

  AssertThrow(dt > 0.0, ExcMessage("FCT requires dt > 0."));


  const auto &M = alpha_consistent_mass_matrix.block(1, 1);

  const auto &D = alpha_diffusion_matrix.block(1, 1);


  // ============================================================
  // Ghosted low-order predictor
  // ============================================================

  alpha_low_relevant = alpha_low_solution.block(1);

  alpha_low_relevant.update_ghost_values();


  // ============================================================
  // Initialize FCT working vectors
  // ============================================================

  fct_P_plus      = 0.0;
  fct_P_minus     = 0.0;
  fct_correction  = 0.0;
  fct_lumped_mass = 0.0;

  fct_R_plus  = 1.0;
  fct_R_minus = 1.0;


  /*
   * TrilinosWrappers::MPI::Vector does not provide the
   * convenient
   *
   *     vector.add(index,value)
   *
   * interface used by some deal.II vector types.
   *
   * Use the indexed add interface instead. This is also
   * suitable for contributions to DoFs owned by another MPI
   * process; compress(VectorOperation::add) communicates them.
   */
  using vector_size_type = TrilinosWrappers::MPI::Vector::size_type;

  const auto add_entry = [](TrilinosWrappers::MPI::Vector &vector,
                            const vector_size_type         index,
                            const double                   value) {
    const vector_size_type indices[1] = {index};

    const double values[1] = {value};

    vector.add(1, indices, values);
  };


  // ============================================================
  // STEP 1:
  // Compute lumped masses
  //
  //     m_i = sum_j M_ij
  // ============================================================

  for (const auto i : alpha_low_solution.block(1).locally_owned_elements())
    {
      double lumped_mass_i = 0.0;

      for (auto entry = M.begin(i); entry != M.end(i); ++entry)
        {
          lumped_mass_i += entry->value();
        }

      AssertThrow(lumped_mass_i > 0.0,
                  ExcMessage("Non-positive lumped alpha mass."));

      fct_lumped_mass[i] = lumped_mass_i;
    }

  fct_lumped_mass.compress(VectorOperation::insert);


  // ============================================================
  // STEP 2:
  // Compute raw pairwise antidiffusive fluxes.
  //
  // Each edge (i,j) is processed exactly ONCE.
  //
  // We define
  //
  //   f_ij =
  //       m_ij (alpha_dot_i - alpha_dot_j)
  //     + d_ij (alpha_i - alpha_j)
  //
  // and enforce
  //
  //   f_ji = -f_ij
  //
  // algebraically rather than calculating it independently.
  // ============================================================

  for (const auto i : alpha_low_solution.block(1).locally_owned_elements())
    {
      const double alpha_i = alpha_low_relevant[i];

      const double alpha_old_i = locally_relevant_old_solution.block(1)[i];

      const double alpha_dot_i = (alpha_i - alpha_old_i) / dt;


      for (auto entry = M.begin(i); entry != M.end(i); ++entry)
        {
          const auto j = entry->column();

          /*
           * Process every algebraic edge exactly once.
           *
           * The diagonal is excluded automatically.
           */
          if (j <= i)
            continue;


          const double m_ij = entry->value();

          const double d_ij = D.el(i, j);


          const double alpha_j = alpha_low_relevant[j];

          const double alpha_old_j = locally_relevant_old_solution.block(1)[j];

          const double alpha_dot_j = (alpha_j - alpha_old_j) / dt;


          double f_ij =
            m_ij * (alpha_dot_i - alpha_dot_j) + d_ij * (alpha_i - alpha_j);

          // Prelimiting drop down-gradient fluxes.
          if (f_ij * (alpha_j - alpha_i) > 0.0)
            f_ij = 0.0;


          /*
           * Since f_ji = -f_ij:
           *
           * If f_ij > 0:
           *
           *   node i receives positive correction
           *   node j receives negative correction.
           *
           * If f_ij < 0, the opposite occurs.
           */
          if (f_ij > 0.0)
            {
              add_entry(fct_P_plus, i, f_ij);

              add_entry(fct_P_minus, j, -f_ij);
            }

          else if (f_ij < 0.0)
            {
              add_entry(fct_P_minus, i, f_ij);

              add_entry(fct_P_plus, j, -f_ij);
            }
        }
    }


  /*
   * MPI communication occurs here, not inside the edge loop.
   */
  fct_P_plus.compress(VectorOperation::add);

  fct_P_minus.compress(VectorOperation::add);


  // ============================================================
  // STEP 3:
  // Compute nodal correction factors R_i^+, R_i^-.
  //
  // Use physical bounds
  //
  //      0 <= alpha <= alpha_max.
  //
  // A tiny safety factor keeps the allowed antidiffusive
  // correction inside the invariant interval in floating-point
  // arithmetic. This modifies the FLUX BUDGET, not alpha.
  // There is no clipping of the solution.
  // ============================================================

  const double eps_safe = 64.0 * std::numeric_limits<double>::epsilon();

  const double safety = 1.0 - eps_safe;


  for (const auto i : alpha_low_solution.block(1).locally_owned_elements())
    {
      const double alpha_i = alpha_low_relevant[i];

      const double lumped_mass_i = fct_lumped_mass[i];


      /*
       * FCT requires the low-order predictor itself
       * to be admissible.
       */
      constexpr double alpha_roundoff_tol = 1e-8;

      AssertThrow(alpha_i >= -alpha_roundoff_tol,
                  ExcMessage("FCT received a negative low-order alpha."));


      AssertThrow(alpha_i <= parameters.alpha_max + alpha_roundoff_tol,
                  ExcMessage("FCT received low-order alpha above alpha_max."));


      const double P_plus = fct_P_plus[i];

      const double P_minus = fct_P_minus[i];


      /*
       * Available upward/downward correction.
       *
       * The safety factor leaves a tiny floating-point
       * margin before the physical bounds.
       */
      const double Q_plus =
        safety * std::max(0.0, parameters.alpha_max - alpha_i);

      const double Q_minus = safety * std::min(0.0, -alpha_i);


      double R_plus  = 1.0;
      double R_minus = 1.0;


      if (P_plus > 0.0)
        {
          R_plus = std::min(1.0, lumped_mass_i * Q_plus / (dt * P_plus));
        }


      if (P_minus < 0.0)
        {
          R_minus = std::min(1.0, lumped_mass_i * Q_minus / (dt * P_minus));
        }


      fct_R_plus[i] = std::max(0.0, R_plus);

      fct_R_minus[i] = std::max(0.0, R_minus);
    }


  fct_R_plus.compress(VectorOperation::insert);

  fct_R_minus.compress(VectorOperation::insert);


  // ============================================================
  // Need limiter values on neighbouring MPI DoFs
  // ============================================================

  fct_R_plus_relevant = fct_R_plus;

  fct_R_minus_relevant = fct_R_minus;

  fct_R_plus_relevant.update_ghost_values();
  fct_R_minus_relevant.update_ghost_values();


  // ============================================================
  // STEP 4:
  // Limit every pairwise flux exactly once.
  //
  // The SAME floating-point number is added to node i and
  // subtracted from node j.
  //
  // Therefore the pair correction is exactly conservative:
  //
  //        F_ij + F_ji = 0.
  // ============================================================

  fct_correction = 0.0;


  for (const auto i : alpha_low_solution.block(1).locally_owned_elements())
    {
      const double alpha_i = alpha_low_relevant[i];

      const double alpha_old_i = locally_relevant_old_solution.block(1)[i];

      const double alpha_dot_i = (alpha_i - alpha_old_i) / dt;


      for (auto entry = M.begin(i); entry != M.end(i); ++entry)
        {
          const auto j = entry->column();

          /*
           * Same unique edge orientation as STEP 2.
           */
          if (j <= i)
            continue;


          const double m_ij = entry->value();

          const double d_ij = D.el(i, j);


          const double alpha_j = alpha_low_relevant[j];

          const double alpha_old_j = locally_relevant_old_solution.block(1)[j];

          const double alpha_dot_j = (alpha_j - alpha_old_j) / dt;


          double f_ij =
            m_ij * (alpha_dot_i - alpha_dot_j) + d_ij * (alpha_i - alpha_j);

          // Prelimiting drop down-gradient fluxes.
          if (f_ij * (alpha_j - alpha_i) > 0.0)
            f_ij = 0.0;


          double limiter = 1.0;


          if (f_ij > 0.0)
            {
              limiter =
                std::min(fct_R_plus_relevant[i], fct_R_minus_relevant[j]);
            }

          else if (f_ij < 0.0)
            {
              limiter =
                std::min(fct_R_minus_relevant[i], fct_R_plus_relevant[j]);
            }


          const double limited_flux = limiter * f_ij;


          /*
           * Exactly conservative bilateral exchange.
           *
           * Do not calculate f_ji independently.
           */
          add_entry(fct_correction, i, limited_flux);

          add_entry(fct_correction, j, -limited_flux);
        }
    }


  fct_correction.compress(VectorOperation::add);


  // ============================================================
  // STEP 5:
  // Apply limited antidiffusive correction
  //
  //      alpha_FCT_i
  //        =
  //      alpha_L_i
  //        +
  //      dt/m_i sum_j F_ij
  //
  // No clipping or projection is performed.
  // ============================================================

  alpha_fct_solution = alpha_low_solution.block(1);


  for (const auto i : alpha_fct_solution.locally_owned_elements())
    {
      const double lumped_mass_i = fct_lumped_mass[i];

      AssertThrow(lumped_mass_i > 0.0,
                  ExcMessage(
                    "Non-positive lumped mass during FCT correction."));

      alpha_fct_solution[i] =
        alpha_low_relevant[i] + dt / lumped_mass_i * fct_correction[i];
    }


  alpha_fct_solution.compress(VectorOperation::insert);
}



template <int dim>
const TrilinosWrappers::MPI::Vector &
SolidPhaseSolver<dim>::get_solid_volume_fraction() const
{
  return locally_relevant_solution.block(1);
}

template <int dim>
const TrilinosWrappers::MPI::Vector &
SolidPhaseSolver<dim>::get_granular_temperature() const
{
  return locally_relevant_theta_solution;
}

template <int dim>
const TrilinosWrappers::MPI::BlockVector &
SolidPhaseSolver<dim>::get_solution() const
{
  return solution;
}


template <int dim>
const TrilinosWrappers::MPI::Vector &
SolidPhaseSolver<dim>::get_theta_solution() const
{
  return theta_solution;
}

template <int dim>
const TrilinosWrappers::MPI::BlockVector &
SolidPhaseSolver<dim>::get_momentum_solution() const
{
  return locally_relevant_momentum_solution;
}



template <int dim>
void
SolidPhaseSolver<dim>::make_output_dir() const
{
  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      std::filesystem::create_directories(parameters.output_folder);
    }
  MPI_Barrier(mpi_communicator);
}


template <int dim>
void
SolidPhaseSolver<dim>::output_results(const double time)
{
  TimerOutput::Scope t(computing_timer, "output");

  if (timestep_number % parameters.output_every != 0)
    return;

  if (parameters.output_verbose)
    {
      pcout << "Writing output..." << std::endl;
    }

  // ------------------------------------------------------------
  // Solid velocity and volume fraction
  // ------------------------------------------------------------
  std::vector<std::string> names(dim, "u");
  names.emplace_back("alpha");

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    interpretation(dim + 1, DataComponentInterpretation::component_is_scalar);

  for (unsigned int d = 0; d < dim; ++d)
    interpretation[d] =
      DataComponentInterpretation::component_is_part_of_vector;


  // ------------------------------------------------------------
  // Single DataOut object for u_s, alpha, and theta
  // ------------------------------------------------------------
  DataOut<dim> data_out;

  data_out.attach_dof_handler(dof_handler);

  data_out.add_data_vector(locally_relevant_solution,
                           names,
                           DataOut<dim>::type_dof_data,
                           interpretation);

  // Granular temperature uses its own DoFHandler, but the same mesh
  data_out.add_data_vector(theta_dof_handler,
                           locally_relevant_theta_solution,
                           "theta");


  // ------------------------------------------------------------
  // MPI subdomain information
  // ------------------------------------------------------------
  Vector<float> subdomain(triangulation.n_active_cells());

  for (unsigned int i = 0; i < subdomain.size(); ++i)
    {
      subdomain(i) =
        static_cast<float>(triangulation.locally_owned_subdomain());
    }

  data_out.add_data_vector(subdomain,
                           "subdomain",
                           DataOut<dim>::type_cell_data);


  // ------------------------------------------------------------
  // Store the actual physical time in the VTU files
  // ------------------------------------------------------------
  DataOutBase::VtkFlags vtk_flags;

  vtk_flags.time  = time;
  vtk_flags.cycle = timestep_number;

  data_out.set_flags(vtk_flags);


  // ------------------------------------------------------------
  // Build and write output
  // ------------------------------------------------------------
  data_out.build_patches();

  const std::string pvtu_filename =
    data_out.write_vtu_with_pvtu_record(parameters.output_folder,
                                        parameters.output_prefix,
                                        timestep_number,
                                        mpi_communicator,
                                        parameters.digits);


  // ------------------------------------------------------------
  // Write PVD file containing the physical time information
  // ------------------------------------------------------------
  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      static std::vector<std::pair<double, std::string>> times_and_names;

      // The .pvd file is inside output_folder, so store only the
      // pvtu filename, not the full directory path.
      const std::string pvtu_basename =
        std::filesystem::path(pvtu_filename).filename().string();

      times_and_names.emplace_back(time, pvtu_basename);

      std::ofstream pvd_output(parameters.output_folder + "/" +
                               parameters.output_prefix + ".pvd");

      DataOutBase::write_pvd_record(pvd_output, times_and_names);
    }
}



template <int dim>
void
SolidPhaseSolver<dim>::setup()
{
  setup_dofs();
  setup_granular_temperature_dofs();

  setup_time_assembler();
  setup_core_assembler();

  make_output_dir();

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar alpha(dim);

  const ComponentMask vel_mask   = fe.component_mask(velocities);
  const ComponentMask alpha_mask = fe.component_mask(alpha);

  const double time = simulation_control->get_current_time();

  auto &velocity_ic_functions = *parameters.initial_conditions.velocity;
  velocity_ic_functions.u.set_time(time);
  velocity_ic_functions.v.set_time(time);
  if constexpr (dim == 3)
    velocity_ic_functions.w.set_time(time);

  auto &alpha_ic_function = *parameters.initial_conditions.alpha;
  alpha_ic_function.set_time(time);

  auto &theta_ic_function = *parameters.initial_conditions.theta;
  theta_ic_function.set_time(time);

  SolidVelocityFunctionDefined<dim> velocity_ic(&velocity_ic_functions.u,
                                                &velocity_ic_functions.v,
                                                (dim == 3 ?
                                                   &velocity_ic_functions.w :
                                                   nullptr));

  SolidAlphaFunctionDefined<dim> alpha_ic(&alpha_ic_function);

  old_solution = 0.0;

  VectorTools::interpolate(dof_handler, velocity_ic, old_solution, vel_mask);
  VectorTools::interpolate(dof_handler, alpha_ic, old_solution, alpha_mask);

  constraints.distribute(old_solution);
  old_solution.compress(VectorOperation::insert);

  older_solution = old_solution;
  older_solution.compress(VectorOperation::insert);

  locally_relevant_old_solution = old_solution;
  locally_relevant_old_solution.update_ghost_values();

  locally_relevant_older_solution = older_solution;
  locally_relevant_older_solution.update_ghost_values();

  solution = old_solution;
  solution.compress(VectorOperation::insert);

  locally_relevant_solution = solution;
  locally_relevant_solution.update_ghost_values();

  locally_relevant_momentum_solution = solution;
  locally_relevant_momentum_solution.update_ghost_values();

  picard_solution = solution;
  picard_solution.compress(VectorOperation::insert);

  locally_relevant_picard_solution = picard_solution;
  locally_relevant_picard_solution.update_ghost_values();

  picard_solution_ptr = &locally_relevant_picard_solution;

  theta_old_solution = 0.0;
  VectorTools::interpolate(theta_dof_handler,
                           theta_ic_function,
                           theta_old_solution);
  theta_constraints.distribute(theta_old_solution);
  theta_old_solution.compress(VectorOperation::insert);

  theta_solution = theta_old_solution;
  theta_solution.compress(VectorOperation::insert);

  theta_picard_solution = theta_solution;
  theta_picard_solution.compress(VectorOperation::insert);

  locally_relevant_theta_old_solution = theta_old_solution;
  locally_relevant_theta_old_solution.update_ghost_values();

  locally_relevant_theta_solution = theta_solution;
  locally_relevant_theta_solution.update_ghost_values();

  locally_relevant_theta_picard_solution = theta_picard_solution;
  locally_relevant_theta_picard_solution.update_ghost_values();

  timestep_number = 0;
}


template <int dim>
void
SolidPhaseSolver<dim>::calculate_mms_error() const
{
  if (!parameters.analytical_solution.enable)
    return;

  if (timestep_number % parameters.analytical_solution.frequency != 0)
    return;

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar alpha(dim);

  const unsigned int n_q =
    std::max<unsigned int>(order + 1, order + parameters.quadrature_extra);

  const QGauss<dim> cell_quadrature(n_q);

  FEValues<dim> fe_values(fe,
                          cell_quadrature,
                          update_values | update_quadrature_points |
                            update_JxW_values);

  const unsigned int n_q_points = cell_quadrature.size();

  std::vector<Tensor<1, dim>> velocity_values(n_q_points);
  std::vector<double>         alpha_values(n_q_points);

  double local_alpha_l2    = 0.0;
  double local_velocity_l2 = 0.0;

  double local_alpha_linf    = 0.0;
  double local_velocity_linf = 0.0;

  const double time = simulation_control->get_current_time();

  parameters.analytical_solution.function->set_time(time);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      fe_values[velocities].get_function_values(locally_relevant_solution,
                                                velocity_values);

      fe_values[alpha].get_function_values(locally_relevant_solution,
                                           alpha_values);

      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          Vector<double> exact(dim + 1);

          parameters.analytical_solution.function->vector_value(
            fe_values.quadrature_point(q), exact);

          Tensor<1, dim> velocity_exact;
          for (unsigned int d = 0; d < dim; ++d)
            velocity_exact[d] = exact[d];

          const double alpha_exact = exact[dim];

          const double JxW = fe_values.JxW(q);

          const double alpha_error = alpha_values[q] - alpha_exact;

          const Tensor<1, dim> velocity_error =
            velocity_values[q] - velocity_exact;

          local_alpha_l2 += alpha_error * alpha_error * JxW;

          local_velocity_l2 += velocity_error.norm_square() * JxW;

          local_alpha_linf = std::max(local_alpha_linf, std::abs(alpha_error));

          local_velocity_linf =
            std::max(local_velocity_linf, velocity_error.norm());
        }
    }

  const double alpha_l2 =
    std::sqrt(Utilities::MPI::sum(local_alpha_l2, mpi_communicator));

  const double velocity_l2 =
    std::sqrt(Utilities::MPI::sum(local_velocity_l2, mpi_communicator));

  const double alpha_linf =
    Utilities::MPI::max(local_alpha_linf, mpi_communicator);

  const double velocity_linf =
    Utilities::MPI::max(local_velocity_linf, mpi_communicator);

  pcout << "solid MMS error:\n"
        << "  alpha L2       = " << alpha_l2 << "\n"
        << "  alpha Linf     = " << alpha_linf << "\n"
        << "  velocity L2    = " << velocity_l2 << "\n"
        << "  velocity Linf  = " << velocity_linf << std::endl;

  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      const std::string filename = parameters.output_folder + "/" +
                                   parameters.analytical_solution.filename;

      const bool write_header = (timestep_number == 1);

      std::ofstream file;

      if (write_header)
        file.open(filename, std::ios::out);
      else
        file.open(filename, std::ios::app);

      if (write_header)
        {
          file << "time step alpha_L2 alpha_Linf velocity_L2 velocity_Linf\n";
        }

      file << std::setprecision(16) << time << " " << timestep_number << " "
           << alpha_l2 << " " << alpha_linf << " " << velocity_l2 << " "
           << velocity_linf << "\n";
    }
}

// Round-off negatives in alpha and Theta: -1e-15 <= v < 0 -> |v|.
// Values below -1e-15 are left as they are.
static void
flip_roundoff_negatives(TrilinosWrappers::MPI::Vector &v,
                        const double                   threshold = 1e-15)
{
  for (const auto i : v.locally_owned_elements())
    {
      const double value = v[i];
      if (value < 0.0 && value >= -threshold)
        v[i] = -value;
    }
  v.compress(VectorOperation::insert);
}

template <int dim>
static void
print_alpha_max_location(const DoFHandler<dim>                    &dof_handler,
                         const TrilinosWrappers::MPI::BlockVector &v,
                         const TrilinosWrappers::MPI::BlockVector &v_old,
                         const MPI_Comm                            comm,
                         const std::string                        &label)
{
  const FiniteElement<dim> &fe    = dof_handler.get_fe();
  const IndexSet            owned = v.locally_owned_elements();

  std::vector<types::global_dof_index> dofs(fe.n_dofs_per_cell());

  double                          local_max = -1.0;
  double                          alpha_old = 0.0;
  Point<dim>                      where;
  std::vector<types::boundary_id> boundary_ids;
  std::vector<Point<dim>>         cell_points;
  std::vector<Tensor<1, dim>>     cell_velocities;
  std::vector<double>             cell_alphas;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      cell->get_dof_indices(dofs);

      for (const unsigned int vtx : cell->vertex_indices())
        {
          const auto a_dof = dofs[fe.component_to_system_index(dim, vtx)];
          if (!owned.is_element(a_dof))
            continue;

          const double a = v(a_dof);
          if (a <= local_max)
            continue;

          local_max = a;
          alpha_old = v_old(a_dof);
          where     = cell->vertex(vtx);

          // Boundary ids of the faces that contain this vertex.
          boundary_ids.clear();
          for (const unsigned int f : cell->face_indices())
            if (cell->face(f)->at_boundary())
              for (const unsigned int fv : cell->face(f)->vertex_indices())
                if (cell->face(f)->vertex_index(fv) == cell->vertex_index(vtx))
                  boundary_ids.push_back(cell->face(f)->boundary_id());

          // alpha and u_s at all vertices of this cell (the node + neighbours).
          cell_points.clear();
          cell_velocities.clear();
          cell_alphas.clear();
          for (const unsigned int w : cell->vertex_indices())
            {
              Tensor<1, dim> u;
              for (unsigned int d = 0; d < dim; ++d)
                u[d] = v(dofs[fe.component_to_system_index(d, w)]);
              cell_points.push_back(cell->vertex(w));
              cell_velocities.push_back(u);
              cell_alphas.push_back(
                v(dofs[fe.component_to_system_index(dim, w)]));
            }
        }
    }

  const double global_max = Utilities::MPI::max(local_max, comm);

  if (local_max == global_max)
    {
      std::ostringstream s;
      s << std::scientific << std::setprecision(4) << label
        << ": alpha = " << local_max << " (old " << alpha_old << ") at x = ("
        << where << "), boundary ids: ";
      if (boundary_ids.empty())
        s << "none (interior node)";
      else
        for (const auto id : boundary_ids)
          s << static_cast<unsigned int>(id) << ' ';
      s << '\n';
      for (unsigned int w = 0; w < cell_points.size(); ++w)
        s << "    cell vertex (" << cell_points[w]
          << "): alpha = " << cell_alphas[w] << ", u_s = ("
          << cell_velocities[w] << ")\n";
      std::cout << s.str() << std::flush;
    }
}


template <int dim>
bool
SolidPhaseSolver<dim>::solve_current_time_step()
{
  const double dt = simulation_control->get_time_step();

  time_assembler->update_coefficients(dt);


  bool converged = false;

  const unsigned int max_picard = parameters.picard_max_iterations;
  const double       picard_tol = parameters.picard_tolerance;
  const double       omega      = parameters.picard_relaxation;

  AssertThrow(omega > 0.0 && omega <= 1.0,
              ExcMessage(
                "Picard relaxation must satisfy 0 < omega <= 1 "
                "for the positivity-preserving projected theta update."));

  TrilinosWrappers::MPI::BlockVector prev_picard(owned_partitioning,
                                                 mpi_communicator);
  TrilinosWrappers::MPI::BlockVector delta(owned_partitioning,
                                           mpi_communicator);

  TrilinosWrappers::MPI::Vector previous_theta(theta_locally_owned,
                                               mpi_communicator);
  TrilinosWrappers::MPI::Vector theta_delta(theta_locally_owned,
                                            mpi_communicator);

  TrilinosWrappers::MPI::Vector solved_theta(theta_locally_owned,
                                             mpi_communicator);

  TrilinosWrappers::MPI::Vector raw_theta_delta(theta_locally_owned,
                                                mpi_communicator);

  TrilinosWrappers::MPI::Vector projected_theta(theta_locally_owned,
                                                mpi_communicator);

  TrilinosWrappers::MPI::Vector projected_theta_delta(theta_locally_owned,
                                                      mpi_communicator);

  // [REVIEW FIX 5] no FCT correction known yet at the first Picard iterate
  locally_relevant_fct_delta = 0.0;

  const auto print_alpha_bounds =
    [&](const TrilinosWrappers::MPI::Vector &alpha, const std::string &label) {
      double local_min = std::numeric_limits<double>::max();
      double local_max = -std::numeric_limits<double>::max();

      for (const auto i : alpha.locally_owned_elements())
        {
          local_min = std::min(local_min, alpha[i]);
          local_max = std::max(local_max, alpha[i]);
        }

      const double global_min =
        Utilities::MPI::min(local_min, mpi_communicator);
      const double global_max =
        Utilities::MPI::max(local_max, mpi_communicator);

      pcout << label << ": min = " << global_min << " max = " << global_max
            << std::endl;
    };

  const auto print_theta_bounds =
    [&](const TrilinosWrappers::MPI::Vector &theta, const std::string &label) {
      double       local_min        = std::numeric_limits<double>::max();
      double       local_max        = -std::numeric_limits<double>::max();
      unsigned int local_non_finite = 0;

      for (const auto i : theta.locally_owned_elements())
        {
          const double value = theta[i];

          if (!std::isfinite(value))
            local_non_finite = 1;
          else
            {
              local_min = std::min(local_min, value);
              local_max = std::max(local_max, value);
            }
        }

      const double global_min =
        Utilities::MPI::min(local_min, mpi_communicator);

      const double global_max =
        Utilities::MPI::max(local_max, mpi_communicator);

      const unsigned int global_non_finite =
        Utilities::MPI::max(local_non_finite, mpi_communicator);

      pcout << label << ": min = " << global_min << " max = " << global_max
            << " non-finite = " << global_non_finite << std::endl;
    };

  for (unsigned int k = 0; k < max_picard; ++k)
    {
      /* Known solid fields alpha^k and u_s^k. */
      picard_solution = solution;
      picard_solution.compress(VectorOperation::insert);

      locally_relevant_picard_solution = picard_solution;
      locally_relevant_picard_solution.update_ghost_values();
      picard_solution_ptr = &locally_relevant_picard_solution;

      prev_picard = solution;
      prev_picard.compress(VectorOperation::insert);

      /* Known granular temperature Theta^k. */
      theta_picard_solution = theta_solution;
      theta_picard_solution.compress(VectorOperation::insert);

      locally_relevant_theta_picard_solution = theta_picard_solution;
      locally_relevant_theta_picard_solution.update_ghost_values();

      previous_theta = theta_solution;
      previous_theta.compress(VectorOperation::insert);

      if (k < 3 || k % 10 == 0)
        print_alpha_bounds(locally_relevant_picard_solution.block(1),
                           "alpha before solve");

      /*
       * Solve the segregated granular-temperature equation.
       *
       * theta_solution initially contains the raw linear solution
       * \hat{Theta}^{k+1}.
       */
      assemble_granular_temperature();
      solve_granular_temperature();

      /*
       * Store the raw theta solution before Picard relaxation.
       */
      solved_theta = theta_solution;
      solved_theta.compress(VectorOperation::insert);


      print_theta_bounds(solved_theta,
                         "raw granular temperature before relaxation");

      raw_theta_delta = solved_theta;
      raw_theta_delta -= previous_theta;



      /*
       * Project the unconstrained theta solution onto Theta >= 0.
       */
      projected_theta = solved_theta;

      flip_roundoff_negatives(projected_theta);

      // for (const auto i : projected_theta.locally_owned_elements())
      //   {
      //     const double theta_value = projected_theta[i];

      //     AssertThrow(std::isfinite(theta_value),
      //                 ExcMessage("Non-finite granular temperature."));

      //     //if (theta_value < 0.0) ++local_clip_count;

      //     const double theta_max = 1.0; // physical cap (m^2/s^2); promote to
      //     a
      //                         // parameter if you prefer.
      //     projected_theta[i] = std::min(theta_max, std::max(0.0,
      //     theta_value));
      //   }

      // projected_theta.compress(VectorOperation::insert);
      // theta_constraints.distribute(projected_theta);
      // projected_theta.compress(VectorOperation::insert);

      const double theta_scale = std::max(1e-30, solved_theta.linfty_norm());
      const double roundoff    = 1e-12 * theta_scale;

      unsigned int local_violations    = 0;
      double       local_most_negative = 0.0;

      for (const auto i : projected_theta.locally_owned_elements())
        {
          const double v = projected_theta[i];

          AssertThrow(std::isfinite(v),
                      ExcMessage("Non-finite granular temperature."));

          local_most_negative = std::min(local_most_negative, v);

          if (v < -roundoff)
            ++local_violations;

          if (v < 0.0)
            projected_theta[i] = 0.0; // round-off only, see check below
        }

      projected_theta.compress(VectorOperation::insert);

      const unsigned int violations =
        Utilities::MPI::sum(local_violations, mpi_communicator);
      const double most_negative =
        Utilities::MPI::min(local_most_negative, mpi_communicator);

      pcout << "theta positivity: most negative raw value = " << most_negative
            << " (round-off bound " << roundoff
            << "), violations = " << violations << std::endl;

      AssertThrow(violations == 0,
                  ExcMessage("Granular temperature negative beyond round-off: "
                             "the M-matrix property of the theta system was "
                             "violated."));



      /*
       * Projected Picard correction.
       */
      projected_theta_delta = projected_theta;
      projected_theta_delta -= previous_theta;


      /*
       * Standard Picard under-relaxation.
       */
      theta_solution = previous_theta;
      theta_solution.add(omega, projected_theta_delta);

      theta_constraints.distribute(theta_solution);
      theta_solution.compress(VectorOperation::insert);

      flip_roundoff_negatives(theta_solution);

      locally_relevant_theta_solution = theta_solution;
      locally_relevant_theta_solution.update_ghost_values();

      if (k < 3 || k % 10 == 0)
        print_theta_bounds(locally_relevant_theta_solution,
                           "granular temperature after relaxation");

      /*
       * Verify that the relaxed theta field is physically admissible.
       */
      double       local_theta_min        = std::numeric_limits<double>::max();
      unsigned int local_theta_non_finite = 0;

      for (const auto i : theta_solution.locally_owned_elements())
        {
          const double theta_value = theta_solution[i];

          if (!std::isfinite(theta_value))
            local_theta_non_finite = 1;
          else
            local_theta_min = std::min(local_theta_min, theta_value);
        }

      const double global_theta_min =
        Utilities::MPI::min(local_theta_min, mpi_communicator);

      const unsigned int global_theta_non_finite =
        Utilities::MPI::max(local_theta_non_finite, mpi_communicator);

      pcout << std::scientific << std::setprecision(3)
            << "global theta minimum = " << global_theta_min << std::endl;



      const unsigned int theta_n_q =
        std::max<unsigned int>(order + 1, order + parameters.quadrature_extra);

      const QGauss<dim> theta_quadrature(theta_n_q);

      FEValues<dim> theta_check_fe_values(theta_fe,
                                          theta_quadrature,
                                          update_values);

      std::vector<double> theta_q_values(theta_quadrature.size());

      double local_theta_q_min = std::numeric_limits<double>::max();

      for (const auto &cell : theta_dof_handler.active_cell_iterators())
        {
          if (!cell->is_locally_owned())
            continue;

          theta_check_fe_values.reinit(cell);

          theta_check_fe_values.get_function_values(
            locally_relevant_theta_solution, theta_q_values);

          for (const double theta_q : theta_q_values)
            {
              AssertThrow(
                std::isfinite(theta_q),
                ExcMessage(
                  "Non-finite post-relaxation theta at quadrature point."));

              local_theta_q_min = std::min(local_theta_q_min, theta_q);
            }
        }

      const double global_theta_q_min =
        Utilities::MPI::min(local_theta_q_min, mpi_communicator);

      pcout << std::scientific << std::setprecision(3)
            << "post-relaxation theta quadrature minimum = "
            << global_theta_q_min << std::endl;

      AssertThrow(global_theta_non_finite == 0,
                  ExcMessage("Relaxed granular-temperature solution contains "
                             "non-finite nodal values."));

      AssertThrow(global_theta_min >= 0.0,
                  ExcMessage("Relaxed granular-temperature solution produced a "
                             "negative nodal value: theta_min = " +
                             std::to_string(global_theta_min)));

      AssertThrow(global_theta_q_min >= 0.0,
                  ExcMessage(
                    "Relaxed granular-temperature field is negative at "
                    "a quadrature point: theta_q_min = " +
                    std::to_string(global_theta_q_min)));

      /* Solve the monolithic continuity-momentum system. */
      assemble_system();

      {
        const auto &L = alpha_low_matrix.block(1, 1);

        double local_min_ratio   = std::numeric_limits<double>::max();
        double local_max_offdiag = -std::numeric_limits<double>::max();

        for (const auto row : L.locally_owned_range_indices())
          {
            double row_sum = 0.0;
            double diag    = 0.0;

            for (auto it = L.begin(row); it != L.end(row); ++it)
              {
                row_sum += it->value();

                if (it->column() == row)
                  diag = it->value();
                else
                  local_max_offdiag = std::max(local_max_offdiag, it->value());
              }

            if (diag != 0.0)
              local_min_ratio = std::min(local_min_ratio, row_sum / diag);
          }

        const double min_ratio =
          Utilities::MPI::min(local_min_ratio, mpi_communicator);
        const double max_offdiag =
          Utilities::MPI::max(local_max_offdiag, mpi_communicator);

        pcout << "alpha low-order M-matrix check: min(rowsum/diag) = "
              << min_ratio << ", max off-diagonal = " << max_offdiag
              << std::endl;

        if (max_offdiag > 0.0)
          pcout << "  Warning: positive off-diagonal in the low-order alpha "
                << "matrix (check the outlet face terms)." << std::endl;

        AssertThrow(min_ratio > 0.0,
                    ExcMessage("Low-order alpha matrix lost diagonal "
                               "dominance: alpha >= 0 is no longer "
                               "guaranteed. Reduce the time step."));
      }

      solve();

      locally_relevant_momentum_solution = solution;
      locally_relevant_momentum_solution.update_ghost_values();

      if (k < 3 || k % 10 == 0)
        print_alpha_bounds(locally_relevant_solution.block(1),
                           "alpha high-order after solve");

      /* Existing relaxation is retained for u_s and alpha only. */
      TrilinosWrappers::MPI::BlockVector solved_solution(owned_partitioning,
                                                         mpi_communicator);
      solved_solution = solution;
      solved_solution.compress(VectorOperation::insert);
      pcout << "alpha low RHS norm = " << alpha_low_rhs.block(1).l2_norm()
            << std::endl;

      pcout << "alpha old norm = " << old_solution.block(1).l2_norm()
            << std::endl;


      /* NEW: solve bounded low-order alpha */
      solve_alpha_low_order();

      flip_roundoff_negatives(alpha_low_solution.block(1));


      double local_min_alpha_low = std::numeric_limits<double>::max();

      for (const auto i : alpha_low_solution.block(1).locally_owned_elements())
        {
          const double alpha_value =
            static_cast<double>(alpha_low_solution.block(1)[i]);

          local_min_alpha_low = std::min(local_min_alpha_low, alpha_value);
        }

      const double global_min_alpha_low =
        Utilities::MPI::min(local_min_alpha_low, mpi_communicator);

      constexpr double alpha_roundoff_tol = 1e-12;

      AssertThrow(global_min_alpha_low >= -alpha_roundoff_tol,
                  ExcMessage("LOW-ORDER positivity failure: alpha_L_min = " +
                             std::to_string(global_min_alpha_low)));



      {
        TrilinosWrappers::MPI::BlockVector alpha_low_ghosted(
          owned_partitioning, relevant_partitioning, mpi_communicator);
        alpha_low_ghosted = alpha_low_solution;
        print_alpha_max_location(dof_handler,
                                 alpha_low_ghosted,
                                 locally_relevant_old_solution,
                                 mpi_communicator,
                                 "alpha low-order max");
      }

      apply_alpha_fct();

      flip_roundoff_negatives(alpha_fct_solution);

      if (k < 3 || k % 10 == 0)
        print_alpha_bounds(alpha_low_solution.block(1), "alpha low-order");

      if (k < 3 || k % 10 == 0)
        print_alpha_bounds(alpha_fct_solution, "alpha FCT");

      double local_min_alpha_fct = std::numeric_limits<double>::max();

      for (const auto i : alpha_fct_solution.locally_owned_elements())
        {
          const double alpha_value = static_cast<double>(alpha_fct_solution[i]);

          local_min_alpha_fct = std::min(local_min_alpha_fct, alpha_value);
        }

      const double global_min_alpha_fct =
        Utilities::MPI::min(local_min_alpha_fct, mpi_communicator);

      AssertThrow(global_min_alpha_fct >= -alpha_roundoff_tol,
                  ExcMessage("FCT positivity failure: alpha_FCT_min = " +
                             std::to_string(global_min_alpha_fct)));


      solved_solution.block(1) = alpha_fct_solution;
      solved_solution.compress(VectorOperation::insert);

      // [REVIEW FIX 5] lagged FCT correction for the next momentum solve:
      // delta = alpha^F - alpha^L (block 0 = 0)
      {
        TrilinosWrappers::MPI::BlockVector delta_owned(owned_partitioning,
                                                       mpi_communicator);
        delta_owned          = 0.0;
        delta_owned.block(1) = alpha_fct_solution;
        delta_owned.block(1) -= alpha_low_solution.block(1);

        locally_relevant_fct_delta = delta_owned;
        locally_relevant_fct_delta.update_ghost_values();
      }

      solution = prev_picard;
      solution *= (1.0 - omega);
      solution.add(omega, solved_solution);

      constraints.distribute(solution);
      solution.compress(VectorOperation::insert);

      flip_roundoff_negatives(solution.block(1));

      locally_relevant_solution = solution;
      locally_relevant_solution.update_ghost_values();

      if (k < 3 || k % 10 == 0)
        print_alpha_bounds(locally_relevant_solution.block(1),
                           "alpha after relaxation");

      delta = solution;
      delta -= prev_picard;

      // theta_delta = theta_solution;
      // theta_delta -= previous_theta;

      // const double solid_picard_error =
      //   delta.l2_norm() / (solution.l2_norm() + 1e-30);

      // const double theta_picard_error =
      //   theta_delta.l2_norm() / (theta_solution.l2_norm() + 1e-30);

      // const double picard_error =
      //   std::max(solid_picard_error, theta_picard_error);

      /*
       * Accepted theta step.
       */
      theta_delta = theta_solution;
      theta_delta -= previous_theta;


      /*
       * Solid Picard error.
       */
      const double solid_picard_error =
        delta.l2_norm() / (omega * (solution.l2_norm() + 1e-30));


      const double velocity_picard_error =
        delta.block(0).l2_norm() /
        (omega * (solution.block(0).l2_norm() + 1e-30));

      const double alpha_picard_error =
        delta.block(1).l2_norm() /
        (omega * (solution.block(1).l2_norm() + 1e-30));

      // pcout << "velocity Picard error = " << velocity_picard_error
      //       << "  alpha Picard error = " << alpha_picard_error << std::endl;


      /*
       * Accepted theta-step error.
       *
       * This is diagnostic only. It must NOT be used as the
       * theta convergence criterion when the positivity line
       * search is active.
       */
      const double theta_accepted_error =
        theta_delta.l2_norm() / (theta_solution.l2_norm() + 1e-30);


      /*
       * Raw fixed-point error.
       *
       * This measures whether the unconstrained theta solve
       * actually satisfies
       *
       *   theta_hat^{k+1} = theta^k.
       *
       * This is the correct quantity to use for Picard
       * convergence.
       */
      const double theta_raw_error =
        raw_theta_delta.l2_norm() / (solved_theta.l2_norm() + 1e-30);

      const double theta_projected_error =
        projected_theta_delta.l2_norm() / (projected_theta.l2_norm() + 1e-30);

      const double theta_picard_error = theta_projected_error;

      const double picard_error =
        std::max(solid_picard_error, theta_picard_error);

      if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0 &&
          parameters.solver_verbose &&
          (k % 5 == 0 || picard_error < picard_tol || k == max_picard - 1))
        {
          pcout << " Picard iteration " << k
                << " solid error = " << solid_picard_error
                << " velocity error = " << velocity_picard_error
                << " alpha error = "
                << alpha_picard_error
                //<< " theta raw error = " << theta_raw_error
                // << " theta projected error = " << theta_projected_error
                // << " theta accepted-step error = " << theta_accepted_error
                << std::endl;
        }

      if (picard_error < picard_tol)
        {
          converged = true;
          break;
        }

      // [REVIEW FIX 9] an unconverged step is not a solution of the scheme;
      // stop instead of accepting it (picard_error is a global norm, so
      // every rank throws).
      AssertThrow(k < max_picard - 1,
                  ExcMessage(
                    "Solid Picard did not converge at t = " +
                    std::to_string(simulation_control->get_current_time()) +
                    ", final error = " + std::to_string(picard_error)));
    }

  return converged;
}

template <int dim>
void
SolidPhaseSolver<dim>::commit_current_time_step()
{
  ++timestep_number;

  const double dt = simulation_control->get_time_step();

  const double time = simulation_control->get_current_time();

  /*
   * Diagnostics must be evaluated before replacing the
   * old-time solution.
   */
  solid_phase_conservation_monitoring();
  // calculate_mms_error();
  output_results(time);

  /*
   * Commit solid velocity and alpha history.
   */
  older_solution = old_solution;
  older_solution.compress(VectorOperation::insert);

  old_solution = solution;
  old_solution.compress(VectorOperation::insert);

  locally_relevant_older_solution = older_solution;
  locally_relevant_older_solution.update_ghost_values();

  locally_relevant_old_solution = old_solution;
  locally_relevant_old_solution.update_ghost_values();

  /*
   * Commit granular-temperature history.
   */
  theta_old_solution = theta_solution;
  theta_old_solution.compress(VectorOperation::insert);

  locally_relevant_theta_old_solution = theta_old_solution;

  locally_relevant_theta_old_solution.update_ghost_values();

  /*
   * Keep your existing CFL calculation here.
   */
  const double max_u_component =
    locally_relevant_solution.block(0).linfty_norm();

  const double max_solid_velocity =
    std::sqrt(static_cast<double>(dim)) * max_u_component;

  double local_h_min = std::numeric_limits<double>::max();

  for (const auto &cell : triangulation.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        local_h_min = std::min(local_h_min, cell->diameter());
    }

  const double h_min = Utilities::MPI::min(local_h_min, mpi_communicator);

  const double solid_cfl = max_solid_velocity * dt / h_min;

  pcout << "solid CFL = " << solid_cfl << "  max|u_s| ~= " << max_solid_velocity
        << "  h_min = " << h_min << "  dt = " << dt << std::endl;

  if (parameters.print_timer_each_step)
    computing_timer.print_summary();
}

template <int dim>
bool
SolidPhaseSolver<dim>::advance_one_step()
{
  const bool converged = solve_current_time_step();

  if (!converged)
    return false;

  commit_current_time_step();

  return true;
}


template <int dim>
void
SolidPhaseSolver<dim>::finalize()
{
  if (parameters.print_timer_at_end)
    {
      computing_timer.print_summary();
    }
}

template <int dim>
bool
SolidPhaseSolver<dim>::finished() const
{
  return simulation_control->is_at_end();
}

template <int dim>
unsigned int
SolidPhaseSolver<dim>::get_step_number() const
{
  return timestep_number;
}

template <int dim>
double
SolidPhaseSolver<dim>::get_current_time() const
{
  return simulation_control->get_current_time();
}


template <int dim>
void
SolidPhaseSolver<dim>::run()
{
  setup();

  while (advance_one_step())
    {
    }

  finalize();
}



template class SolidPhaseSolver<2>;
template class SolidPhaseSolver<3>;