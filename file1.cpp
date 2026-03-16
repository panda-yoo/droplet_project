
#include <iomanip>
#include <iostream>
#include <map>
#include <random>

// Random number engine
std::random_device rd;
std::mt19937 gen(rd());

// Lambda (rate) = 1.0 (mean time = 1.0)
std::exponential_distribution<double> expd(1e-2);
std::uniform_real_distribution<double> unid(1.0, 0.0);
// Generate and display some random numbers

const double Lx = 100.0;
const double Ly = 100.0;

double getdistance(double x1, double x2, double y1, double y2) {

  // double getdistance(Particle &p1, Particle &p2) {

  double d2 = ((x1 - x2) * (x1 - x2)) + ((y1 - y2) * (y1 - y2));

  return std::sqrt(d2);
}
class Particle {
public:
  Particle() {

    px_ = Lx * unid(gen);
    py_ = Ly * unid(gen);

    px0_ = px_;
    py0_ = py_;
  }

  void evolve() {

    double u = unid(gen);
    double step = -std::log(u) / lambda;

    double theta = 2.0 * M_PI * unid(gen);

    px_ += step * std::cos(theta);
    py_ += step * std::sin(theta);
  }

  double getx() const { return px_; }
  double gety() const { return py_; }

  double getdistance() const {

    double dx = px_ - px0_;
    double dy = py_ - py0_;

    return std::sqrt(dx * dx + dy * dy);
  }

private:
  double px0_, py0_; // initial position
  double px_, py_;   // current position
};

int main() { return 0; }