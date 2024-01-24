
// NIX MODULE TO READ ALL PARAMETERS FROM nix_params.yaml

struct PathParams 
{
    string nix;
    string read;
    string out;
};


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


struct ConstantsParams 
{
    double g;
    double rho;
    double rho_w;
    double sec_year;
};

struct BedrockEwsParams 
{
    string smooth;
    int p;
    double sigma_gauss;
    double t0_smth;
    double x_1;
    double x_2;
    double y_p;
    double y_0;
};

struct DomainParams 
{
    string exp;
    int n;
    int n_z;
    string grid;
    double grid_exp;
    BedrockEwsParams ews;
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
    bool stoch;
    double t0;
    int N;
    double dt_noise;
};

struct SMBParams 
{
    bool stoch;
    double t0_stoch;
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


struct TrendParams 
{
    string type;
    double t0;
    double tf;
    double M_0;
    double M_f;
};

struct BoundaryConditionsParams 
{
    SMBParams smb;
    ThermParams therm;
    TrendParams trend;
};

struct AdvWParams 
{
    bool apply;
    double w_0;
    double w_exp;
};

struct ThermodynamicsParams 
{
    bool therm;
    double k;
    double G_k;
    double kappa;
    double theta_max;
    double theta_act;
    AdvWParams adv_w;
};

struct FrictionParams 
{
    double m;
    double u_0;
    string therm;
    double theta_thw;
    double theta_frz;
    double C_ref_0;
    double C_frz;
    double C_thw;
    double C_ews;
};

struct ViscosityParams 
{
    double n_gln;
    double n_exp;
    double eps;
    bool therm;
    double t_eq_A_theta;
    double A_act;
    double Q_act_1;
    double Q_act_2;
    double A_0_1;
    double A_0_2;
    double R;
};

struct AdvectionParams 
{
    string meth;
};


struct SubShelfMeltParams 
{
    bool shelf_melt;
    string meth;
    double t0_oce;
    double tf_oce;
    double c_po;
    double L_i;
    double gamma_T;
    double T_0;
};

struct CalvingParams 
{
    string meth;
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
    double visc_0;
    double theta;
    double beta;
};

struct NixParams 
{
    PathParams path;
    TimeParams tm;
    ConstantsParams cnst;
    DomainParams dom;
    DynamicsParams dyn;
    TimestepParams tmstep;
    StochasticParams stoch; 
    BoundaryConditionsParams bc;
    ThermodynamicsParams thrmdyn;
    FrictionParams fric;
    ViscosityParams vis;
    AdvectionParams adv;
    CalvingParams calv;
    PicardParams pcrd;
    InitParams init;
};


// Here we populate the structures and transform in the desried units.
void readParams(const YAML::Node& node, NixParams& params) 
{
    // PATHS.
    params.path.nix  = node["path"]["nix"].as<string>();
    params.path.out  = node["path"]["out"].as<string>();
    params.path.read = node["path"]["read"].as<string>();
    
    //TIME.
    params.tm.t0            = node["time"]["t0"].as<double>();
    params.tm.tf            = node["time"]["tf"].as<double>();
    params.tm.t_eq          = node["time"]["t_eq"].as<double>();
    params.tm.output.t_n    = node["time"]["output"]["t_n"].as<int>();
    params.tm.output.out_hr = node["time"]["output"]["out_hr"].as<bool>();

    // PHYSICAL CONSTANTS.
    params.cnst.g        = node["constants"]["g"].as<double>();
    params.cnst.rho      = node["constants"]["rho"].as<double>();
    params.cnst.rho_w    = node["constants"]["rho_w"].as<double>();
    params.cnst.sec_year = node["constants"]["sec_year"].as<double>();

    // DOMAIN.
    params.dom.exp             = node["domain"]["experiment"].as<std::string>();
    params.dom.n               = node["domain"]["n"].as<int>();
    params.dom.n_z             = node["domain"]["n_z"].as<int>();
    params.dom.grid            = node["domain"]["grid"].as<std::string>();
    params.dom.grid_exp        = node["domain"]["grid_exp"].as<double>();
    params.dom.ews.smooth      = node["domain"]["bedrock_ews"]["smooth_bed"].as<string>();
    params.dom.ews.p           = node["domain"]["bedrock_ews"]["p"].as<int>();
    params.dom.ews.sigma_gauss = node["domain"]["bedrock_ews"]["sigma_gauss"].as<double>();
    params.dom.ews.t0_smth    = node["domain"]["bedrock_ews"]["t0_smth"].as<double>();
    params.dom.ews.x_1         = node["domain"]["bedrock_ews"]["x_1"].as<double>();
    params.dom.ews.x_2         = node["domain"]["bedrock_ews"]["x_2"].as<double>();
    params.dom.ews.y_p         = node["domain"]["bedrock_ews"]["y_p"].as<double>();
    params.dom.ews.y_0         = node["domain"]["bedrock_ews"]["y_0"].as<double>();

    // DYNAMICS.
    params.dyn.vel_meth = node["dynamics"]["vel_meth"].as<std::string>();

    // TIME STEP.
    params.tmstep.dt_meth = node["timestep"]["dt_meth"].as<std::string>();
    params.tmstep.dt_min  = node["timestep"]["dt_min"].as<double>();
    params.tmstep.dt_max  = node["timestep"]["dt_max"].as<double>();
    params.tmstep.t_eq_dt = node["timestep"]["t_eq_dt"].as<double>();
    params.tmstep.rel     = node["timestep"]["rel"].as<double>();

    // STOCHASIC BOUNDARY CONDITIONS.
    params.stoch.stoch    = node["stochastic"]["stoch"].as<bool>();
    params.stoch.t0       = node["stochastic"]["t0"].as<double>();
    params.stoch.N        = node["stochastic"]["N"].as<int>();
    params.stoch.dt_noise = node["stochastic"]["dt_noise"].as<double>();

    // DETERMINISTIC BOUNDARY CONDITIONS.
    params.bc.smb.S_0      = node["boundary_conditions"]["smb"]["S_0"].as<double>();
    params.bc.smb.dlta_smb = node["boundary_conditions"]["smb"]["dlta_smb"].as<double>();
    params.bc.smb.x_acc    = node["boundary_conditions"]["smb"]["x_acc"].as<double>();
    params.bc.smb.x_mid    = node["boundary_conditions"]["smb"]["x_mid"].as<double>();
    params.bc.smb.x_sca    = node["boundary_conditions"]["smb"]["x_sca"].as<double>();
    params.bc.smb.x_varmid = node["boundary_conditions"]["smb"]["x_varmid"].as<double>();
    params.bc.smb.x_varsca = node["boundary_conditions"]["smb"]["x_varsca"].as<double>();
    params.bc.smb.var_mult = node["boundary_conditions"]["smb"]["var_mult"].as<double>();
    params.bc.trend.type   = node["boundary_conditions"]["trend"]["type"].as<std::string>();
    params.bc.trend.t0     = node["boundary_conditions"]["trend"]["t0"].as<double>();
    params.bc.trend.tf     = node["boundary_conditions"]["trend"]["tf"].as<double>();
    params.bc.trend.M_0    = node["boundary_conditions"]["trend"]["M_0"].as<double>();
    params.bc.trend.M_f    = node["boundary_conditions"]["trend"]["M_f"].as<double>();
    params.bc.therm.T_air  = node["boundary_conditions"]["therm"]["T_air"].as<double>();
    params.bc.therm.w_min  = node["boundary_conditions"]["therm"]["w_min"].as<double>();
    params.bc.therm.w_max  = node["boundary_conditions"]["therm"]["w_max"].as<double>();

    // THERMODYNAMICS.
    params.thrmdyn.therm       = node["thermodynamics"]["therm"].as<bool>();
    params.thrmdyn.k           = node["thermodynamics"]["k"].as<double>();
    params.thrmdyn.G_k         = node["thermodynamics"]["G"].as<double>() / params.thrmdyn.k;
    params.thrmdyn.kappa       = params.cnst.sec_year * node["thermodynamics"]["kappa"].as<double>();
    params.thrmdyn.theta_max   = node["thermodynamics"]["theta_max"].as<double>();
    params.thrmdyn.theta_act   = node["thermodynamics"]["theta_act"].as<double>();
    params.thrmdyn.adv_w.apply = node["thermodynamics"]["adv_w"]["apply"].as<bool>();
    params.thrmdyn.adv_w.w_0   = node["thermodynamics"]["adv_w"]["w_0"].as<double>();
    params.thrmdyn.adv_w.w_exp = node["thermodynamics"]["adv_w"]["w_exp"].as<double>();

    
    // FRICTION.
    params.fric.m          = node["friction"]["m"].as<double>();
    params.fric.u_0        = node["friction"]["u_0"].as<double>();
    params.fric.therm      = node["friction"]["therm"].as<string>();
    params.fric.theta_frz  = node["friction"]["theta_frz"].as<double>();
    params.fric.theta_thw  = node["friction"]["theta_thw"].as<double>();
    params.fric.C_ref_0    = node["friction"]["C_ref"].as<double>() / pow(params.cnst.sec_year, params.fric.m);
    params.fric.C_frz      = node["friction"]["C_frz"].as<double>() / pow(params.cnst.sec_year, params.fric.m);
    params.fric.C_ews      = node["friction"]["C_ews"].as<double>() / pow(params.cnst.sec_year, params.fric.m);
    params.fric.C_thw      = params.fric.C_frz * node["friction"]["C_thw"].as<double>();

    // VISCOSITY.
    params.vis.n_gln        = node["viscosity"]["n_gln"].as<double>();
    params.vis.n_exp        = ( 1.0 - params.vis.n_gln ) / ( 2.0 * params.vis.n_gln );
    params.vis.eps          = node["viscosity"]["eps"].as<double>();
    params.vis.therm        = node["viscosity"]["therm"].as<bool>();
    params.vis.t_eq_A_theta = node["viscosity"]["t_eq_A_theta"].as<double>();
    params.vis.A_act        = params.cnst.sec_year * node["viscosity"]["A_act"].as<double>();
    params.vis.Q_act_1      = 1.0e3 * node["viscosity"]["Q_act_1"].as<double>();
    params.vis.Q_act_2      = 1.0e3 * node["viscosity"]["Q_act_2"].as<double>();
    params.vis.A_0_1        = params.cnst.sec_year * node["viscosity"]["A_0_1"].as<double>();
    params.vis.A_0_2        = params.cnst.sec_year * node["viscosity"]["A_0_2"].as<double>();
    params.vis.R            = node["viscosity"]["R"].as<double>();

    // HORIZONTAL ICE ADVECTION.
    params.adv.meth = node["advection"]["meth"].as<string>();

    // CALVING AND SUH-SHELF MELT.
    params.calv.meth                      = node["calving"]["meth"].as<string>();
    params.calv.m_dot                     = node["calving"]["m_dot"].as<double>();
    params.calv.sub_shelf_melt.shelf_melt = node["calving"]["sub_shelf_melt"]["shelf_melt"].as<bool>();
    params.calv.sub_shelf_melt.meth       = node["calving"]["sub_shelf_melt"]["meth"].as<string>();
    params.calv.sub_shelf_melt.t0_oce     = node["calving"]["sub_shelf_melt"]["t0_oce"].as<double>();
    params.calv.sub_shelf_melt.tf_oce     = node["calving"]["sub_shelf_melt"]["tf_oce"].as<double>();
    params.calv.sub_shelf_melt.c_po       = node["calving"]["sub_shelf_melt"]["c_po"].as<double>();
    params.calv.sub_shelf_melt.L_i        = node["calving"]["sub_shelf_melt"]["L_i"].as<double>();
    params.calv.sub_shelf_melt.gamma_T    = params.cnst.sec_year * node["calving"]["sub_shelf_melt"]["gamma_T"].as<double>();
    params.calv.sub_shelf_melt.T_0        = node["calving"]["sub_shelf_melt"]["T_0"].as<double>();

    // PICARD ITERATION.
    params.pcrd.n       = node["picard"]["n"].as<int>();
    params.pcrd.tol     = node["picard"]["tol"].as<double>();
    params.pcrd.omega_1 = M_PI * node["picard"]["omega_1"].as<double>();
    params.pcrd.omega_2 = M_PI * node["picard"]["omega_2"].as<double>();

    // INTIALISATION.
    params.init.H      = node["initialisation"]["H"].as<double>();
    params.init.S      = node["initialisation"]["S"].as<double>();
    params.init.u      = node["initialisation"]["u"].as<double>();
    params.init.visc_0 = node["initialisation"]["visc_0"].as<double>();
    params.init.theta  = node["initialisation"]["theta"].as<double>();
    params.init.beta   = node["initialisation"]["beta"].as<double>();
}