/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  num_particles = 1000;

  default_random_engine gen;

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; i++)
  {
    double px = dist_x(gen);
    double py = dist_y(gen);
    double pt = dist_theta(gen);

    Particle particle;
    particle.id = i;
    particle.x = px;
    particle.y = py;
    particle.theta = pt;
    particle.weight = 1.0 / num_particles;

    particles.push_back(particle);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  default_random_engine gen;

  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for (int i = 0; i < num_particles; i ++)  {
    double x = particles[i].x;
    double y = particles[i].y;
    double theta = particles[i].theta;

    if (fabs(yaw_rate) > 1E-4) {
      x += velocity  / yaw_rate * (sin(theta + yaw_rate * delta_t) - sin(theta));
      y += velocity  / yaw_rate * (cos(theta) - cos(theta + yaw_rate*delta_t));
    }
    else {
      x += velocity * delta_t * cos(theta);
      y += velocity * delta_t * sin(theta);
    }
    theta += yaw_rate * delta_t;

    // add random noise
    x += dist_x(gen);
    y += dist_y(gen);
    theta += dist_theta(gen);

    // Update particle coordinates
    particles[i].x = x;
    particles[i].y = y;
    particles[i].theta = theta;

  }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

  for (int i = 0; i < observations.size(); i++)
  {
    double ox = observations[i].x;
    double oy = observations[i].y;

    double min_distance = INT32_MAX;  // some big number

    for (int j = 0; j < predicted.size(); j++) {
      double dx = predicted[j].x - ox;
      double dy = predicted[j].y - oy;
      double dist = dx * dx + dy * dy;
      if (dist < min_distance)
      {
        min_distance = dist;
        observations[i].id = i;
      }
    }
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  // non-constant version of the vector
  std::vector<LandmarkObs> obs = observations;


  double w_sum = 0;

  for (int i = 0; i < num_particles; i++)
  {

    // Predict map landmarks in car's coordinate system

    std::vector<LandmarkObs> predictions;
    for (int j = 0; j < map_landmarks.landmark_list.size(); j++)
    {
      double lx = map_landmarks.landmark_list[j].x_f;
      double ly = map_landmarks.landmark_list[j].y_f;
      double lid = map_landmarks.landmark_list[j].id_i;

      // translate
      double px = lx - particles[i].x;
      double py = ly - particles[i].y;

      // rotate
      double rtheta = -particles[i].theta;   //

      LandmarkObs prediction = LandmarkObs();
      prediction.id = lid;
      prediction.x = px * cos(rtheta) + py * sin(rtheta);
      prediction.y = -px * sin(rtheta) + py * cos(rtheta);
      predictions.push_back(prediction);
    }

    double weight = 1.0;

    // find nearest observations

    dataAssociation(predictions, obs);

    double sig_x = std_landmark[0];
    double sig_y = std_landmark[1];

    for (int j = 0; j < obs.size(); j++) {
      LandmarkObs observation = obs[j];
      LandmarkObs prediction = predictions[obs[j].id];

      double dx = observation.x - prediction.x;
      double dy = observation.y - prediction.y;
      double distance = sqrt(dx * dx + dy * dy);

      if (distance > sensor_range)
      {
        // Landmark is too far. Scale it to sensor range
        dx *= sensor_range / distance;
        dy *= sensor_range / distance;
      }

      // Calculate Multivariate-Gaussian probability density
      double multi_prob = exp( -(dx * dx / (2 * sig_x * sig_x) + dy*dy / (2 * sig_y * sig_y))) / (2 * M_PI * sig_x * sig_y);

      weight *= multi_prob;
    }
    particles[i].weight = weight;

    w_sum += weight; // We will use it for normalization

  }

  // normalize weights
  for (int i = 0; i < num_particles; i++)
  {
    particles[i].weight /= w_sum;
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
