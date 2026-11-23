#include <stdexcept>
#include <algorithm>
#include "sdnn/backend.hpp"

namespace sdnn {

namespace{
// Returns true if any neuron in right_layer is nonzero after the bias and clamp
bool apply_activations(const std::vector<float>& left_layer,
						     std::vector<float>& right_layer,
					   const Dataset& d,
					   int left_layer_num){

	std::fill(right_layer.begin(), right_layer.end(), 0.0f);

	std::span<const std::int32_t> indptr  = d.layer_indptr(left_layer_num);
	std::span<const std::int32_t> indices = d.layer_indices(left_layer_num);
	std::span<const float>        values  = d.layer_values(left_layer_num);

	// For every neuron in left layer, this adds it's contribution (edge_weight * neuron_value) 
	// to each neuron it is connected to in the right layer
	for(std::int32_t i = 0; i < d.neurons; ++i){
		if(left_layer[i] == 0.0f){
			continue;
		}		
		std::int32_t row_start = indptr[i];
		std::int32_t row_end = indptr[i+1];
		for(std::int32_t k = row_start; k < row_end; ++k){
			right_layer[indices[k]] += values[k] * left_layer[i];
		}
	}

	bool alive = false;
	for(std::int32_t i = 0; i < d.neurons; ++i){
		if(right_layer[i] > 0) right_layer[i] += d.bias;
		right_layer[i] = std::min(std::max(0.0f, right_layer[i]), d.ymax);
		alive |= (right_layer[i] > 0.0f);
	}
	return alive;
}
}

std::vector<std::int32_t> infer_openmp(const Dataset &d, int layers) {
	std::vector<std::int32_t> is_image_active(d.images.rows, 0);
	std::vector<std::int32_t> active_image_idxs;
	#pragma omp parallel
	{
	
		// Per thread activation buffers. Reused by each image allocated to that thread
		std::vector<float> left_layer(d.neurons, 0.0f);
		std::vector<float> right_layer(d.neurons, 0.0f);

		#pragma omp for
		for(std::size_t image = 0; image < d.images.rows; ++image){
			std::int32_t image_start = d.images.indptr[image];
			std::int32_t image_end = d.images.indptr[image + 1];

			std::fill(left_layer.begin(), left_layer.end(), 0.0f);

			for(auto i = image_start; i < image_end; ++i){
				left_layer[d.images.indices[i]] = d.images.values[i];
			}

			for(int i = 0; i < layers; ++i){
				bool alive = apply_activations(left_layer, right_layer, d, i);
				std::swap(left_layer, right_layer);
				if(!alive) break;
			}

			bool nonzero = false;
			for(std::int32_t i = 0; i < d.neurons; ++i){
				if(left_layer[i] != 0.0f){
					nonzero = true;
					break;
				}
			}

			is_image_active[image] = nonzero;
		}
	}

	for(std::size_t image = 0; image < d.images.rows; ++image){
		if(is_image_active[image]) active_image_idxs.push_back(image);
	}

	return active_image_idxs;
}

} // namespace sdnn
