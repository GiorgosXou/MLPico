int get_next_bit(void) {
    static unsigned int value = 0;
    static int bit_pos = 2;

    int bit = (value >> bit_pos) & 1;

    bit_pos--;

    if (bit_pos < 0) {
        bit_pos = 2;
        value++;
    }

    return bit;
}
