
// NIX MODULE TO READ ALL PARAMETERS FROM nix_params.yaml

struct OutputParams 
{
    int t_n;
    bool out_hr;
};


struct TimeParams 
{
    double t0;
    double tf;
    double t_eq;
    OutputParams output;
};


struct ConstantsParams {
    double g;
    double rho;
    double rho_w;
    double sec_year;
};

struct BedrockEwsParams 
{
    int smooth_bed;
    double sigma_gauss;
    double t0_gauss;
    double x_1;
    double x_2;
    double y_p;
    double y_0;
};

struct DomainParams {
    string experiment;
    int n;
    int n_z;
    string grid;
    double grid_exp;
    BedrockEwsParams bedrock_ews;
};

struct DynamicsParams 
{
    string vel_meth;
};

struct TimestepParams 
{
    string dt_meth;
    double dt_min;
    double dt_max;
    double t_eq_dt;
    double rel;
};

struct StochasticParams 
{
    int N;
    double dt_noise;
};

struct SMBParams 
{
    bool stoch;
    double S_0;
    double dlta_smb;
    double x_acc;
    double x_mid;
    double x_sca;
    double x_varmid;
    double x_varsca;
    double var_mult;
};

struct ThermParams 
{
    double T_air;
    double w_min;
    double w_max;
};

struct BoundaryConditionsParams 
{
    SMBParams smb;
    ThermParams therm;
};

struct ThermodynamicsParams 
{
    bool therm;
    int therm_w;
    double k;
    double G;
    double kappa;
    double theta_max;
    double theta_act;
    double R;
};

struct FrictionParams 
{
    double m;
    double u_0;
    int fric_therm;
    double theta_frz;
    double C_ref;
    double C_frz;
    double C_thw;
};

struct ViscosityParams 
{
    double n_gln;
    double eps;
    bool visc_therm;
    double t_eq_A_theta;
    double A_act;
    double visc_0;
    double visc_min;
    double visc_max;
    double Q_act_1;
    double Q_act_2;
    double A_0_1;
    double A_0_2;
};

struct AdvectionParams 
{
    int meth;
};


struct SubShelfMeltParams 
{
    bool shelf;
    int meth;
    double t0_oce;
    double tf_oce;
    double c_po;
    double L_i;
    double gamma_T;
    double T_0;
    double delta_T_oce;
    double M;
    double delta_M;
};

struct CalvingParams 
{
    int meth;
    double m_dot;
    SubShelfMeltParams sub_shelf_melt;
};

struct PicardParams 
{
    int n;
    double tol;
    double omega_1;
    double omega_2;
};

struct InitParams 
{
    double H;
    double S;
    double u;
    double theta;
    double beta;
};

struct NixParams 
{
    TimeParams time;
    ConstantsParams constants;
    DomainParams domain;
    DynamicsParams dynamics;
    TimestepParams timestep;
    StochasticParams stochastic;
    BoundaryConditionsParams boundary_conditions;
    ThermodynamicsParams thermodynamics;
    FrictionParams friction;
    ViscosityParams viscosity;
    AdvectionParams advection;
    CalvingParams calving;
    PicardParams picard;
    InitParams initialisation;
};


void readParams(const YAML::Node& node, NixParams& params) 
{
    params.time.t0 = node["time"]["t0"].as<double>();
    params.time.tf = node["time"]["tf"].as<double>();
    params.time.t_eq = node["time"]["t_eq"].as<double>();
    params.time.output.t_n = node["time"]["output"]["t_n"].as<int>();
    params.time.output.out_hr = node["time"]["output"]["out_hr"].as<bool>();

    params.constants.g = node["constants"]["g"].as<double>();
    params.constants.rho = node["constants"]["rho"].as<double>();
    params.constants.rho_w = node["constants"]["rho_w"].as<double>();
    params.constants.sec_year = node["constants"]["sec_year"].as<double>();

    params.domain.experiment = node["domain"]["experiment"].as<std::string>();
    params.domain.n = node["domain"]["n"].as<int>();
    params.domain.n_z = node["domain"]["n_z"].as<int>();
    params.domain.grid = node["domain"]["grid"].as<std::string>();
    params.domain.grid_exp = node["domain"]["grid_exp"].as<double>();
    params.domain.bedrock_ews.smooth_bed = node["domain"]["bedrock_ews"]["smooth_bed"].as<int>();
    params.domain.bedrock_ews.sigma_gauss = node["domain"]["bedrock_ews"]["sigma_gauss"].as<double>();
    params.domain.bedrock_ews.t0_gauss = node["domain"]["bedrock_ews"]["t0_gauss"].as<double>();
    params.domain.bedrock_ews.x_1 = node["domain"]["bedrock_ews"]["x_1"].as<double>();
    params.domain.bedrock_ews.x_2 = node["domain"]["bedrock_ews"]["x_2"].as<double>();
    params.domain.bedrock_ews.y_p = node["domain"]["bedrock_ews"]["y_p"].as<double>();
    params.domain.bedrock_ews.y_0 = node["domain"]["bedrock_ews"]["y_0"].as<double>();

    params.dynamics.vel_meth = node["dynamics"]["vel_meth"].as<std::string>();

    params.timestep.dt_meth = node["timestep"]["dt_meth"].as<std::string>();
    params.timestep.dt_min = node["timestep"]["dt_min"].as<double>();
    params.timestep.dt_max = node["timestep"]["dt_max"].as<double>();
    params.timestep.t_eq_dt = node["timestep"]["t_eq_dt"].as<double>();
    params.timestep.rel = node["timestep"]["rel"].as<double>();

    params.stochastic.N = node["stochastic"]["N"].as<int>();
    params.stochastic.dt_noise = node["stochastic"]["dt_noise"].as<double>();

    params.boundary_conditions.smb.stoch = node["boundary_conditions"]["smb"]["stoch"].as<bool>();
    params.boundary_conditions.smb.S_0 = node["boundary_conditions"]["smb"]["S_0"].as<double>();
    params.boundary_conditions.smb.dlta_smb = node["boundary_conditions"]["smb"]["dlta_smb"].as<double>();
    params.boundary_conditions.smb.x_acc = node["boundary_conditions"]["smb"]["x_acc"].as<double>();
    params.boundary_conditions.smb.x_mid = node["boundary_conditions"]["smb"]["x_mid"].as<double>();
    params.boundary_conditions.smb.x_sca = node["boundary_conditions"]["smb"]["x_sca"].as<double>();
    params.boundary_conditions.smb.x_varmid = node["boundary_conditions"]["smb"]["x_varmid"].as<double>();
    params.boundary_conditions.smb.x_varsca = node["boundary_conditions"]["smb"]["x_varsca"].as<double>();
    params.boundary_conditions.smb.var_mult = node["boundary_conditions"]["smb"]["var_mult"].as<double>();

    params.boundary_conditions.therm.T_air = node["boundary_conditions"]["therm"]["T_air"].as<double>();
    params.boundary_conditions.therm.w_min = node["boundary_conditions"]["therm"]["w_min"].as<double>();
    params.boundary_conditions.therm.w_max = node["boundary_conditions"]["therm"]["w_max"].as<double>();

    params.thermodynamics.therm = node["thermodynamics"]["therm"].as<bool>();
    params.thermodynamics.therm_w = node["thermodynamics"]["therm_w"].as<int>();
    params.thermodynamics.k = node["thermodynamics"]["k"].as<double>();
    params.thermodynamics.G = node["thermodynamics"]["G"].as<double>();
    params.thermodynamics.kappa = node["thermodynamics"]["kappa"].as<double>();
    params.thermodynamics.theta_max = node["thermodynamics"]["theta_max"].as<double>();
    params.thermodynamics.theta_act = node["thermodynamics"]["theta_act"].as<double>();
    params.thermodynamics.R = node["thermodynamics"]["R"].as<double>();

    params.friction.m = node["friction"]["m"].as<double>();
    params.friction.u_0 = node["friction"]["u_0"].as<double>();
    params.friction.fric_therm = node["friction"]["fric_therm"].as<int>();
    params.friction.theta_frz = node["friction"]["theta_frz"].as<double>();
    params.friction.C_ref = node["friction"]["C_ref"].as<double>();
    params.friction.C_frz = node["friction"]["C_frz"].as<double>();
    params.friction.C_thw = node["friction"]["C_thw"].as<double>();

    params.viscosity.n_gln = node["viscosity"]["n_gln"].as<double>();
    params.viscosity.eps = node["viscosity"]["eps"].as<double>();
    params.viscosity.visc_therm = node["viscosity"]["visc_therm"].as<bool>();
    params.viscosity.t_eq_A_theta = node["viscosity"]["t_eq_A_theta"].as<double>();
    params.viscosity.A_act = node["viscosity"]["A_act"].as<double>();
    params.viscosity.visc_0 = node["viscosity"]["visc_0"].as<double>();
    params.viscosity.visc_min = node["viscosity"]["visc_min"].as<double>();
    params.viscosity.visc_max = node["viscosity"]["visc_max"].as<double>();
    params.viscosity.Q_act_1 = node["viscosity"]["Q_act_1"].as<double>();
    params.viscosity.Q_act_2 = node["viscosity"]["Q_act_2"].as<double>();
    params.viscosity.A_0_1 = node["viscosity"]["A_0_1"].as<double>();
    params.viscosity.A_0_2 = node["viscosity"]["A_0_2"].as<double>();

    params.advection.meth = node["advection"]["meth"].as<int>();

    params.calving.meth = node["calving"]["meth"].as<int>();
    params.calving.m_dot = node["calving"]["m_dot"].as<double>();
    params.calving.sub_shelf_melt.shelf = node["calving"]["sub_shelf_melt"]["shelf"].as<bool>();
    params.calving.sub_shelf_melt.meth = node["calving"]["sub_shelf_melt"]["meth"].as<int>();
    params.calving.sub_shelf_melt.t0_oce = node["calving"]["sub_shelf_melt"]["t0_oce"].as<double>();
    params.calving.sub_shelf_melt.tf_oce = node["calving"]["sub_shelf_melt"]["tf_oce"].as<double>();
    params.calving.sub_shelf_melt.c_po = node["calving"]["sub_shelf_melt"]["c_po"].as<double>();
    params.calving.sub_shelf_melt.L_i = node["calving"]["sub_shelf_melt"]["L_i"].as<double>();
    params.calving.sub_shelf_melt.gamma_T = node["calving"]["sub_shelf_melt"]["gamma_T"].as<double>();
    params.calving.sub_shelf_melt.T_0 = node["calving"]["sub_shelf_melt"]["T_0"].as<double>();
    params.calving.sub_shelf_melt.delta_T_oce = node["calving"]["sub_shelf_melt"]["delta_T_oce"].as<double>();
    params.calving.sub_shelf_melt.M = node["calving"]["sub_shelf_melt"]["M"].as<double>();
    params.calving.sub_shelf_melt.delta_M = node["calving"]["sub_shelf_melt"]["delta_M"].as<double>();

    params.picard.n = node["picard"]["n"].as<int>();
    params.picard.tol = node["picard"]["tol"].as<double>();
    params.picard.omega_1 = node["picard"]["omega_1"].as<double>();
    params.picard.omega_2 = node["picard"]["omega_2"].as<double>();

    params.initialisation.H = node["initialisation"]["H"].as<double>();
    params.initialisation.S = node["initialisation"]["S"].as<double>();
    params.initialisation.u = node["initialisation"]["u"].as<double>();
    params.initialisation.theta = node["initialisation"]["theta"].as<double>();
    params.initialisation.beta = node["initialisation"]["beta"].as<double>();
}