// ObjectModel native core: the kinematics that run inside estimation and control loops.
//
// Mirrors objectmodel/geometry.py, objectmodel/joints.py and the IK step in objectmodel/humanoid.py
// so results can be compared one-to-one. Exposed through a plain C ABI (no Python headers needed):
// loaded with ctypes by objectmodel/native.py. All matrices are row-major doubles.
//
// Build: cmake -S cpp -B cpp/build && cmake --build cpp/build   (needs Eigen 3)

#include <Eigen/Dense>
#include <chrono>
#include <cmath>

using Mat3 = Eigen::Matrix3d;
using Vec3 = Eigen::Vector3d;
using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

namespace {

constexpr int kOk = 0;
constexpr int kBadArgs = 1;
constexpr double kEps = 1e-12;

// Rotation R and translation t of motion k from a packed array [R(9), t(3)] * T.
inline Mat3 rot_of(const double* motions, int k) {
  return Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(motions + 12 * k);
}
inline Vec3 trans_of(const double* motions, int k) { return Eigen::Map<const Vec3>(motions + 12 * k + 9); }

}  // namespace

extern "C" {

int om_version() { return 1; }

// Weighted Kabsch: R, t minimising sum w_i |R p_i + t - q_i|^2. w may be null (uniform).
int om_kabsch(const double* P, const double* Q, const double* w, int n, double* R_out, double* t_out) {
  if (!P || !Q || !R_out || !t_out || n < 3) return kBadArgs;
  Eigen::Map<const RowMat> p(P, n, 3), q(Q, n, 3);
  Eigen::VectorXd wt = Eigen::VectorXd::Ones(n);
  if (w) wt = Eigen::Map<const Eigen::VectorXd>(w, n);
  const double ws = wt.sum();
  if (ws <= kEps) return kBadArgs;
  wt /= ws;
  const Eigen::RowVector3d cp = wt.transpose() * p;
  const Eigen::RowVector3d cq = wt.transpose() * q;
  const RowMat pc = p.rowwise() - cp;
  const RowMat qc = q.rowwise() - cq;
  const Mat3 H = pc.transpose() * wt.asDiagonal() * qc;
  Eigen::JacobiSVD<Mat3> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Mat3 U = svd.matrixU(), V = svd.matrixV();
  double d = (V * U.transpose()).determinant() < 0 ? -1.0 : 1.0;
  Mat3 D = Mat3::Identity();
  D(2, 2) = d;
  const Mat3 R = V * D * U.transpose();
  const Vec3 t = cq.transpose() - R * cp.transpose();
  { Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> out_map(R_out); out_map = R; }
  { Eigen::Map<Vec3> out_map(t_out); out_map = t; }
  return kOk;
}

// Revolute joint from T relative motions: axis = null vector of stacked (R_k - I); pivot from
// least squares on (I - R_k) p = t_k, moved to the axis point nearest `centroid`; signed angles,
// oriented so the largest swing is positive, then unwrapped.
int om_fit_revolute(const double* motions, int T, const double* centroid, double* axis_out, double* pivot_out,
                    double* theta_out) {
  if (!motions || !centroid || !axis_out || !pivot_out || !theta_out || T < 1) return kBadArgs;
  RowMat A(3 * T, 3), M(3 * T, 3);
  Eigen::VectorXd b(3 * T);
  for (int k = 0; k < T; ++k) {
    const Mat3 R = rot_of(motions, k);
    A.block<3, 3>(3 * k, 0) = R - Mat3::Identity();
    M.block<3, 3>(3 * k, 0) = Mat3::Identity() - R;
    b.segment<3>(3 * k) = trans_of(motions, k);
  }
  Eigen::JacobiSVD<RowMat> svdA(A, Eigen::ComputeThinV);
  Vec3 u = svdA.matrixV().col(2);
  // minimum-norm least squares, same as numpy.linalg.lstsq
  Vec3 p = M.completeOrthogonalDecomposition().solve(b);
  const Vec3 c = Eigen::Map<const Vec3>(centroid);
  p += (c - p).dot(u) * u;
  Eigen::VectorXd th(T);
  for (int k = 0; k < T; ++k) {
    const Mat3 R = rot_of(motions, k);
    const Vec3 vee(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));
    th(k) = std::atan2(0.5 * u.dot(vee), 0.5 * (R.trace() - 1.0));
  }
  Eigen::Index imax = 0;
  th.cwiseAbs().maxCoeff(&imax);
  if (th(imax) < 0) {
    u = -u;
    th = -th;
  }
  for (int k = 1; k < T; ++k) {  // unwrap
    double dlt = th(k) - th(k - 1);
    while (dlt > M_PI) { th(k) -= 2 * M_PI; dlt -= 2 * M_PI; }
    while (dlt < -M_PI) { th(k) += 2 * M_PI; dlt += 2 * M_PI; }
  }
  { Eigen::Map<Vec3> out_map(axis_out); out_map = u; }
  { Eigen::Map<Vec3> out_map(pivot_out); out_map = p; }
  { Eigen::Map<Eigen::VectorXd> out_map(theta_out, T); out_map = th; }
  return kOk;
}

// Prismatic joint: axis = top right singular vector of the stacked translations; d_k = t_k . axis.
int om_fit_prismatic(const double* motions, int T, double* axis_out, double* d_out) {
  if (!motions || !axis_out || !d_out || T < 1) return kBadArgs;
  RowMat Tm(T, 3);
  for (int k = 0; k < T; ++k) Tm.row(k) = trans_of(motions, k).transpose();
  Eigen::JacobiSVD<RowMat> svd(Tm, Eigen::ComputeThinV);
  Vec3 u = svd.matrixV().col(0);
  Eigen::VectorXd d = Tm * u;
  Eigen::Index imax = 0;
  d.cwiseAbs().maxCoeff(&imax);
  if (d(imax) < 0) {
    u = -u;
    d = -d;
  }
  { Eigen::Map<Vec3> out_map(axis_out); out_map = u; }
  { Eigen::Map<Eigen::VectorXd> out_map(d_out, T); out_map = d; }
  return kOk;
}

// One damped-least-squares step: dq = J^T (J J^T + lambda^2 I)^-1 err, J is 3 x n.
int om_dls_step(const double* J, int n, const double* err, double lambda, double* dq_out) {
  if (!J || !err || !dq_out || n < 1) return kBadArgs;
  Eigen::Map<const RowMat> Jm(J, 3, n);
  const Vec3 e = Eigen::Map<const Vec3>(err);
  const Mat3 JJt = Jm * Jm.transpose() + lambda * lambda * Mat3::Identity();
  const Vec3 y = JJt.ldlt().solve(e);
  { Eigen::Map<Eigen::VectorXd> out_map(dq_out, n); out_map = Jm.transpose() * y; }
  return kOk;
}

// Time n DLS steps inside C++ (no FFI overhead); returns nanoseconds per step.
double om_bench_dls(const double* J, int n, const double* err, double lambda, int iters) {
  if (!J || !err || n < 1 || iters < 1) return -1.0;
  Eigen::VectorXd dq(n);
  volatile double sink = 0.0;
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    om_dls_step(J, n, err, lambda, dq.data());
    sink = sink + dq(0);
  }
  const auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;
}

}  // extern "C"
