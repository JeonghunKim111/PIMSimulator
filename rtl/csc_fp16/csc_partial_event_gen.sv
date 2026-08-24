module csc_partial_event_gen (
  input logic clk, input logic rst_n,
  input logic mul_result_valid, output logic mul_result_ready,
  input logic [255:0] mul_result, input logic [511:0] row_indices,
  input logic [15:0] lane_mask,
  output logic event_valid, input logic event_ready,
  output logic [31:0] event_row_idx, output logic [15:0] event_value
);
  logic [3:0] lane;
  logic active;
  logic [255:0] result_hold;
  logic [511:0] rows_hold;
  logic [15:0] mask_hold;
  assign mul_result_ready=!active;
  assign event_valid=active && mask_hold[lane];
  assign event_row_idx=rows_hold[lane*32 +: 32];
  assign event_value=result_hold[lane*16 +: 16];
  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin active<=0; lane<=0;result_hold<='0;rows_hold<='0;mask_hold<='0;end
    else begin
      if(mul_result_valid && mul_result_ready) begin
        active<=1;lane<=0;result_hold<=mul_result;rows_hold<=row_indices;mask_hold<=lane_mask;
      end
      else if(active) begin
        if(!mask_hold[lane] || event_ready) begin
          if(lane==15) begin active<=0; lane<=0; end else lane<=lane+1'b1;
        end
      end
    end
  end
endmodule
