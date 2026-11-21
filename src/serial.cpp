#include <stdexcept>
#include <algorithm>

#include "sdnn/backend.hpp"

namespace sdnn {

namespace{
void apply_activations(const std::vector<float>& left_layer,
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

	for(std::int32_t i = 0; i < d.neurons; ++i){
		if(right_layer[i] > 0) right_layer[i] += d.bias;
		right_layer[i] = std::min(std::max(0.0f, right_layer[i]), d.ymax);
	}
}
}

std::vector<std::int32_t> infer_serial(const Dataset &d, int layers) {
	std::vector<std::int32_t> active_image_idxs;

	for(std::size_t image = 0; image < d.images.rows; ++image){
		std::int32_t image_start = d.images.indptr[image];
		std::int32_t image_end = d.images.indptr[image + 1];

		// First layer is the image itself
		std::vector<float> left_layer(d.neurons, 0.0f);
		for(auto i = image_start; i < image_end; ++i){
			left_layer[d.images.indices[i]] = d.images.values[i];
		}

		for(int i = 0; i < layers; ++i){
			std::vector<float> right_layer(d.neurons, 0.0f);
			apply_activations(left_layer, right_layer, d, i);
			std::swap(left_layer, right_layer);
		}

		// Now, left_layer points to the last layer of the neural net
		bool nonzero = false;
		for(std::int32_t i = 0; i < d.neurons; ++i){
			if(left_layer[i] != 0.0f){
				nonzero = true;
				break;
			}
		}

		if(nonzero){
			active_image_idxs.push_back(image);
		}
	}

	return active_image_idxs;
}
} // namespace sdnn
