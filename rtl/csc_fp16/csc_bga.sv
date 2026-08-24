module csc_bga #(
  parameter int ACC_ENTRIES=64, parameter int COMPARE_W=64,
  parameter int INPUT_DEPTH=64, parameter int OUTPUT_DEPTH=64,
  parameter int ROW_W=32, parameter int VALUE_W=16, parameter int BATCH_WIDTH=8
) (
  input logic clk, input logic rst_n,
  input logic in_valid, output logic in_ready,
  input logic [ROW_W-1:0] in_row, input logic [VALUE_W-1:0] in_value,
  input logic drain_req, output logic drain_done,
  output logic add_req_valid, input logic add_req_ready,
  output logic [VALUE_W-1:0] add_a, output logic [VALUE_W-1:0] add_b,
  input logic add_result_valid, input logic [VALUE_W-1:0] add_result,
  output logic out_valid, input logic out_ready,
  output logic [ROW_W-1:0] out_row, output logic [VALUE_W-1:0] out_value,
  output logic [$clog2(ACC_ENTRIES+1)-1:0] occupancy
);
  localparam int AIW=(ACC_ENTRIES<=1)?1:$clog2(ACC_ENTRIES);
  localparam int IIW=(INPUT_DEPTH<=1)?1:$clog2(INPUT_DEPTH);
  localparam int OIW=(OUTPUT_DEPTH<=1)?1:$clog2(OUTPUT_DEPTH);
  localparam int OCW=$clog2(OUTPUT_DEPTH+1);
  localparam int ICW=$clog2(INPUT_DEPTH+1);
  localparam int AGE_W=(ACC_ENTRIES<=1)?1:$clog2(ACC_ENTRIES);
  logic [ROW_W-1:0] inq_row[INPUT_DEPTH];
  logic [VALUE_W-1:0] inq_val[INPUT_DEPTH];
  logic [IIW-1:0] in_rd,in_wr; logic [ICW-1:0] in_count;
  logic [ROW_W-1:0] tags[ACC_ENTRIES];
  logic [VALUE_W-1:0] sums[ACC_ENTRIES];
  logic [AGE_W-1:0] ages[ACC_ENTRIES];
  logic [ACC_ENTRIES-1:0] valid;
  logic [ROW_W-1:0] outq_row[OUTPUT_DEPTH];
  logic [VALUE_W-1:0] outq_val[OUTPUT_DEPTH];
  logic [OIW-1:0] out_rd,out_wr; logic [OCW-1:0] out_count;
  logic hit,free_exists; logic [AIW-1:0] hit_idx,free_idx,victim_idx;
  logic [AGE_W-1:0] min_age;
  logic waiting_add; logic [AIW-1:0] add_idx;
  logic draining;
  integer i; integer j; integer valid_count_int;
  logic service_pop, service_push_out;
  logic [ROW_W-1:0] service_out_row;
  logic [VALUE_W-1:0] service_out_val;

  initial begin
    if(COMPARE_W>ACC_ENTRIES) $error("COMPARE_W must be <= ACC_ENTRIES");
  end
  assign in_ready=(in_count<INPUT_DEPTH);
  assign out_valid=(out_count!=0);
  assign out_row=outq_row[out_rd]; assign out_value=outq_val[out_rd];
  assign add_a=sums[hit_idx]; assign add_b=inq_val[in_rd];
  assign add_req_valid=!waiting_add && !draining && in_count!=0 && hit;
  always_comb begin
    hit=0; free_exists=0; hit_idx='0; free_idx='0; victim_idx='0;
    min_age={AGE_W{1'b1}}; valid_count_int=0;
    for(i=0;i<ACC_ENTRIES;i=i+1) begin
      if(valid[i]) begin
        valid_count_int=valid_count_int+1;
        if(i<COMPARE_W && tags[i]==inq_row[in_rd] && !hit) begin hit=1; hit_idx=i[AIW-1:0]; end
        if(ages[i]<min_age) begin min_age=ages[i]; victim_idx=i[AIW-1:0]; end
      end else if(!free_exists) begin free_exists=1; free_idx=i[AIW-1:0]; end
    end
    occupancy=valid_count_int;
  end
  always_comb begin
    service_pop=0; service_push_out=0; service_out_row='0; service_out_val='0;
    if(!waiting_add && !draining && in_count!=0) begin
      if(hit) begin service_pop=add_req_valid && add_req_ready; end
      else if(free_exists) service_pop=1;
      else if(out_count<OUTPUT_DEPTH || (out_valid&&out_ready)) begin
        service_pop=1; service_push_out=1;
        service_out_row=tags[victim_idx]; service_out_val=sums[victim_idx];
      end
    end else if(draining && valid_count_int!=0 &&
                (out_count<OUTPUT_DEPTH || (out_valid&&out_ready))) begin
      service_push_out=1; service_out_row=tags[victim_idx]; service_out_val=sums[victim_idx];
    end
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
      in_rd<=0;in_wr<=0;in_count<=0;out_rd<=0;out_wr<=0;out_count<=0;
      valid<='0;waiting_add<=0;add_idx<=0;draining<=0;drain_done<=0;
    end else begin
      drain_done<=0;
      if(drain_req && !draining) draining<=1;
      if(in_valid && in_ready) begin
        inq_row[in_wr]<=in_row; inq_val[in_wr]<=in_value;
        in_wr<=(in_wr==INPUT_DEPTH-1)?0:in_wr+1'b1;
      end
      if(service_pop) in_rd<=(in_rd==INPUT_DEPTH-1)?0:in_rd+1'b1;
      case({in_valid&&in_ready,service_pop})
        2'b10: in_count<=in_count+1'b1;
        2'b01: in_count<=in_count-1'b1;
        default: begin end
      endcase
      if(add_req_valid && add_req_ready) begin waiting_add<=1; add_idx<=hit_idx; end
      if(waiting_add && add_result_valid) begin sums[add_idx]<=add_result; waiting_add<=0; end
      if(service_pop && !hit) begin
        if(free_exists) begin
          valid[free_idx]<=1; tags[free_idx]<=inq_row[in_rd]; sums[free_idx]<=inq_val[in_rd];
          ages[free_idx]<=valid_count_int[AGE_W-1:0];
        end else begin
          tags[victim_idx]<=inq_row[in_rd]; sums[victim_idx]<=inq_val[in_rd];
          ages[victim_idx]<=ACC_ENTRIES-1;
          for(j=0;j<ACC_ENTRIES;j=j+1) if(valid[j] && j!=victim_idx && ages[j]!=0) ages[j]<=ages[j]-1'b1;
        end
      end
      if(draining && service_push_out) begin
        valid[victim_idx]<=0;
        for(j=0;j<ACC_ENTRIES;j=j+1) if(valid[j] && j!=victim_idx && ages[j]!=0) ages[j]<=ages[j]-1'b1;
      end
      if(service_push_out) begin
        outq_row[out_wr]<=service_out_row; outq_val[out_wr]<=service_out_val;
        out_wr<=(out_wr==OUTPUT_DEPTH-1)?0:out_wr+1'b1;
      end
      if(out_valid&&out_ready) out_rd<=(out_rd==OUTPUT_DEPTH-1)?0:out_rd+1'b1;
      case({service_push_out,out_valid&&out_ready})
        2'b10: out_count<=out_count+1'b1;
        2'b01: out_count<=out_count-1'b1;
        default: begin end
      endcase
      if(draining && valid_count_int==0 && !waiting_add && in_count==0 && out_count==0) begin
        draining<=0; drain_done<=1;
      end
    end
  end
endmodule
