module csc_batch_ingress #(
  parameter int BATCH_WIDTH=8, parameter int ROW_W=32, parameter int VALUE_W=16
) (
  input logic clk, input logic rst_n,
  input logic batch_valid, output logic batch_ready,
  input logic [$clog2(BATCH_WIDTH+1)-1:0] batch_count,
  input logic [BATCH_WIDTH*ROW_W-1:0] batch_rows,
  input logic [BATCH_WIDTH*VALUE_W-1:0] batch_values,
  output logic event_valid, input logic event_ready,
  output logic [ROW_W-1:0] event_row, output logic [VALUE_W-1:0] event_value
);
  localparam int CW=$clog2(BATCH_WIDTH+1);
  logic active; logic [CW-1:0] count, index;
  logic [BATCH_WIDTH*ROW_W-1:0] rows;
  logic [BATCH_WIDTH*VALUE_W-1:0] values;
  assign batch_ready=!active;
  assign event_valid=active && index<count;
  assign event_row=rows[index*ROW_W +: ROW_W];
  assign event_value=values[index*VALUE_W +: VALUE_W];
  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin active<=0; count<=0; index<=0; rows<='0; values<='0; end
    else begin
      if(batch_valid && batch_ready) begin
        rows<=batch_rows; values<=batch_values; count<=batch_count; index<=0;
        active<=batch_count!=0;
      end else if(event_valid && event_ready) begin
        if(index+1>=count) begin active<=0; index<=0; end
        else index<=index+1'b1;
      end
    end
  end
endmodule
