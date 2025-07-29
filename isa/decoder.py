from typing import Dict, Any, Optional
from .arm_instructions import ARMInstructionDecoder


class InstructionDecoder:
    """
    Main instruction decoder for ARM CPU simulator
    
    Provides high-level instruction decoding interface that integrates
    with the CPU execution pipeline.
    """
    
    def __init__(self):
        """Initialize instruction decoder"""
        self.decoder = ARMInstructionDecoder()
        self.decode_cache = {}  # Cache for frequently decoded instructions
        self.cache_hits = 0
        self.cache_misses = 0
    
    def decode(self, instruction: int, address: int = 0) -> Dict[str, Any]:
        """
        Decode ARM instruction
        
        Args:
            instruction: 32-bit instruction word
            address: Address of instruction (for context)
            
        Returns:
            Decoded instruction dictionary
        """
        # Check decode cache first
        if instruction in self.decode_cache:
            self.cache_hits += 1
            cached_result = self.decode_cache[instruction].copy()
            cached_result['address'] = address
            return cached_result
        
        # Decode instruction
        self.cache_misses += 1
        decoded = self.decoder.decode_instruction(instruction)
        decoded['address'] = address
        
        # Cache the result (without address)
        cache_entry = decoded.copy()
        cache_entry.pop('address', None)
        self.decode_cache[instruction] = cache_entry
        
        # Limit cache size
        if len(self.decode_cache) > 1000:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self.decode_cache.keys())[:100]
            for key in keys_to_remove:
                del self.decode_cache[key]
        
        return decoded
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get decoder cache statistics"""
        total_accesses = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(total_accesses, 1)
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.decode_cache)
        }
    
    def clear_cache(self):
        """Clear decode cache"""
        self.decode_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0


# Example usage
if __name__ == "__main__":
    decoder = InstructionDecoder()
    
    # Test instruction decoding
    # MOV R0, #42
    mov_instr = 0xE3A0002A
    decoded = decoder.decode(mov_instr, 0x8000)
    print(f"Decoded MOV: {decoded}")
    
    # Test cache
    decoded_again = decoder.decode(mov_instr, 0x8004)
    stats = decoder.get_cache_statistics()
    print(f"Cache stats: {stats}")
