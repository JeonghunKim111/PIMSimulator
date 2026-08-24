module csc_descriptor_engine #(
  parameter int LANES = 16
) (
  input  logic clk, input logic rst_n,
  input  logic desc_valid, output logic desc_ready,
  input  logic [31:0] desc_column_id, input logic [31:0] desc_x_slot,
  input  logic [31:0] desc_nnz_count,
  input  logic [63:0] desc_value_offset, input logic [63:0] desc_index_offset,
  output logic chunk_valid, input logic chunk_ready,
  output logic [31:0] column_id, output logic [31:0] x_slot,
  output logic [63:0] value_cursor, output logic [63:0] index_cursor,
  output logic [15:0] lane_mask, output logic [4:0] valid_lanes,
  output logic [31:0] chunk_id, output logic descriptor_done, output logic busy
);
  logic [31:0] remaining;
  always_comb begin
    desc_ready = !busy;
    chunk_valid = busy;
    valid_lanes = (remaining >= LANES) ? LANES[4:0] : remaining[4:0];
    lane_mask = (remaining >= LANES) ? 16'hffff : (16'hffff >> (16-valid_lanes));
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      busy <= 1'b0; remaining <= '0; column_id <= '0; x_slot <= '0;
      value_cursor <= '0; index_cursor <= '0; chunk_id <= '0;
      descriptor_done <= 1'b0;
    end else begin
      descriptor_done <= 1'b0;
      if (desc_valid && desc_ready) begin
        column_id <= desc_column_id; x_slot <= desc_x_slot;
        remaining <= desc_nnz_count; value_cursor <= desc_value_offset;
        index_cursor <= desc_index_offset; chunk_id <= 0;
        busy <= (desc_nnz_count != 0);
        if (desc_nnz_count == 0) descriptor_done <= 1'b1;
      end else if (chunk_valid && chunk_ready) begin
        if (remaining <= LANES) begin
          remaining <= 0; busy <= 1'b0; descriptor_done <= 1'b1;
        end else begin
          remaining <= remaining - LANES;
          value_cursor <= value_cursor + 32;
          index_cursor <= index_cursor + (LANES * 4);
          chunk_id <= chunk_id + 1;
        end
      end
    end
  end
endmodule
