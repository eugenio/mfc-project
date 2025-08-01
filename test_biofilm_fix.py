#!/usr/bin/env python3
"""
Quick test to verify BiofilmKineticsModel fix
"""
import sys
import os
sys.path.insert(0, os.path.join('q-learning-mfcs', 'src'))

from biofilm_kinetics import BiofilmKineticsModel

def test_model_fix():
    """Test the BiofilmKineticsModel fix"""
    try:
        print("🧪 Testing BiofilmKineticsModel fix...")
        
        # Test model initialization
        model = BiofilmKineticsModel(species='geobacter', substrate='acetate', use_gpu=False)
        print("✅ Model initialization successful")
        
        # Test kinetic_params attribute
        if hasattr(model, 'kinetic_params'):
            print("✅ kinetic_params exists")
            print(f"   mu_max: {model.kinetic_params.mu_max}")
        else:
            print("❌ kinetic_params missing")
            return False
            
        # Test substrate_props attribute  
        if hasattr(model, 'substrate_props'):
            print("✅ substrate_props exists")
            print(f"   molecular_weight: {model.substrate_props.molecular_weight}")
        else:
            print("❌ substrate_props missing")
            return False
            
        # Test get_model_parameters method
        try:
            params = model.get_model_parameters()
            print("✅ get_model_parameters() works")
            print(f"   kinetic_params keys: {list(params['kinetic_params'].keys())}")
        except Exception as e:
            print(f"❌ get_model_parameters() failed: {e}")
            return False
            
        # Test calculate_theoretical_maximum_current method
        try:
            max_current = model.calculate_theoretical_maximum_current()
            print("✅ calculate_theoretical_maximum_current() works")
            print(f"   max_current: {max_current}")
        except Exception as e:
            print(f"❌ calculate_theoretical_maximum_current() failed: {e}")
            return False
            
        print("🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_fix()
    exit(0 if success else 1)