import cv2
import numpy as np
import os
import time

class ScreenLayoutManager:
    """
    Manages different screen layouts for testing customer attention.
    Provides structured grid layouts with labeled regions and analytics.
    """
    
    def __init__(self, base_width=1920, base_height=1080):
        """Initialize the screen layout manager with default dimensions."""
        self.base_width = base_width
        self.base_height = base_height
        self.current_layout = "default"
        self.current_layout_index = 0
        
        # Directory for saving analytics
        self.output_dir = "output/screen_analytics"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Track regions of interest for analytics
        self.regions = {}
        
        # Initialize available layouts
        self.layouts = {
            "default": self._create_default_layout,
            "grid": self._create_grid_layout,
            "product_comparison": self._create_product_comparison_layout,
            "checkout": self._create_checkout_layout
        }
        
        # Default blank screen
        self.current_screen = self._create_blank_screen()

    def _create_blank_screen(self):
        """Create a blank white screen with the specified dimensions."""
        return np.ones((self.base_height, self.base_width, 3), dtype=np.uint8) * 255
    
    def _add_region(self, name, x1, y1, x2, y2, color=None):
        """Add a region of interest to the current layout."""
        self.regions[name] = {
            "coords": (x1, y1, x2, y2),
            "color": color,
            "gaze_points": 0,
            "total_time": 0,
            "last_visit": None
        }
    
    def create_layout(self, layout_name=None):
        """Create and return a screen with the specified layout."""
        # If no layout name provided, use current layout
        if layout_name is None:
            layout_name = self.current_layout
        
        # Reset regions
        self.regions = {}
        
        # Create the specified layout
        if layout_name in self.layouts:
            self.current_layout = layout_name
            self.current_screen = self.layouts[layout_name]()
            return self.current_screen
        else:
            print(f"Warning: Layout '{layout_name}' not found. Using default.")
            self.current_layout = "default"
            self.current_screen = self.layouts["default"]()
            return self.current_screen
    
    def next_layout(self):
        """Switch to the next available layout."""
        layout_names = list(self.layouts.keys())
        self.current_layout_index = (self.current_layout_index + 1) % len(layout_names)
        self.current_layout = layout_names[self.current_layout_index]
        self.current_screen = self.create_layout(self.current_layout)
        print(f"Switched to layout: {self.current_layout}")
        return self.current_screen
    
    def _create_default_layout(self):
        """Create the default layout (similar to original)."""
        screen = self._create_blank_screen()
        
        # Header
        cv2.rectangle(screen, (0, 0), (self.base_width, 100), (200, 200, 200), -1)
        cv2.putText(screen, "Customer Attention Analysis", (self.base_width//2 - 200, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
        self._add_region("header", 0, 0, self.base_width, 100, (200, 200, 200))
        
        # Left sidebar
        cv2.rectangle(screen, (0, 100), (300, self.base_height), (230, 230, 230), -1)
        self._add_region("sidebar", 0, 100, 300, self.base_height, (230, 230, 230))
        
        # Product A
        cv2.rectangle(screen, (400, 200), (700, 500), (0, 0, 255), 2)
        cv2.putText(screen, "Product A", (500, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        self._add_region("product_a", 400, 200, 700, 500, (0, 0, 255))
        
        # Product B
        cv2.rectangle(screen, (800, 200), (1100, 500), (0, 255, 0), 2)
        cv2.putText(screen, "Product B", (900, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        self._add_region("product_b", 800, 200, 1100, 500, (0, 255, 0))
        
        # Product C
        cv2.rectangle(screen, (1200, 200), (1500, 500), (255, 0, 0), 2)
        cv2.putText(screen, "Product C", (1300, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        self._add_region("product_c", 1200, 200, 1500, 500, (255, 0, 0))
        
        # Buttons
        cv2.rectangle(screen, (400, 600), (700, 700), (100, 100, 200), -1)
        cv2.putText(screen, "Buy Now", (500, 660), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        self._add_region("buy_now", 400, 600, 700, 700, (100, 100, 200))
        
        cv2.rectangle(screen, (800, 600), (1100, 700), (100, 200, 100), -1)
        cv2.putText(screen, "Add to Cart", (850, 660), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        self._add_region("add_to_cart", 800, 600, 1100, 700, (100, 200, 100))
        
        cv2.rectangle(screen, (1200, 600), (1500, 700), (200, 100, 100), -1)
        cv2.putText(screen, "More Info", (1300, 660), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        self._add_region("more_info", 1200, 600, 1500, 700, (200, 100, 100))
        
        return screen
    
    def _create_grid_layout(self):
        """Create a grid layout with labeled zones A through I."""
        screen = self._create_blank_screen()
        
        # Header
        cv2.rectangle(screen, (0, 0), (self.base_width, 100), (200, 200, 200), -1)
        cv2.putText(screen, "Grid Layout Testing", (self.base_width//2 - 150, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
        self._add_region("header", 0, 0, self.base_width, 100, (200, 200, 200))
        
        # Create a 3x3 grid
        cell_width = (self.base_width - 100) // 3
        cell_height = (self.base_height - 200) // 3
        
        zones = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        colors = [
            (255, 200, 200), (200, 255, 200), (200, 200, 255),
            (255, 255, 200), (255, 200, 255), (200, 255, 255),
            (240, 240, 240), (220, 220, 220), (200, 200, 200)
        ]
        
        for i in range(3):
            for j in range(3):
                idx = i * 3 + j
                zone = zones[idx]
                color = colors[idx]
                
                x1 = 50 + j * cell_width
                y1 = 150 + i * cell_height
                x2 = x1 + cell_width
                y2 = y1 + cell_height
                
                # Draw cell
                cv2.rectangle(screen, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(screen, (x1, y1), (x2, y2), (0, 0, 0), 2)
                
                # Add zone label
                cv2.putText(screen, f"Zone {zone}", (x1 + cell_width//2 - 50, y1 + cell_height//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
                
                # Add region
                self._add_region(f"zone_{zone.lower()}", x1, y1, x2, y2, color)
        
        return screen
    
    def _create_product_comparison_layout(self):
        """Create a layout for product comparison with categorized sections."""
        screen = self._create_blank_screen()
        
        # Header
        cv2.rectangle(screen, (0, 0), (self.base_width, 100), (200, 200, 200), -1)
        cv2.putText(screen, "Product Comparison", (self.base_width//2 - 150, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
        self._add_region("header", 0, 0, self.base_width, 100, (200, 200, 200))
        
        # Product columns
        col_width = (self.base_width - 400) // 3
        products = ["Budget Option", "Popular Choice", "Premium Model"]
        product_colors = [(200, 230, 255), (255, 230, 200), (230, 255, 200)]
        
        # Category rows
        categories = ["Image", "Price", "Features", "Reviews", "Availability"]
        row_heights = [300, 100, 200, 150, 100]
        category_colors = [(240, 240, 240), (245, 245, 245), (250, 250, 250), (245, 245, 245), (240, 240, 240)]
        
        # Draw product headers
        for i, product in enumerate(products):
            x1 = 200 + i * col_width
            y1 = 100
            x2 = x1 + col_width
            y2 = y1 + 50
            
            cv2.rectangle(screen, (x1, y1), (x2, y2), product_colors[i], -1)
            cv2.rectangle(screen, (x1, y1), (x2, y2), (0, 0, 0), 1)
            cv2.putText(screen, product, (x1 + 20, y1 + 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            self._add_region(f"product_{i+1}_header", x1, y1, x2, y2, product_colors[i])
        
        # Draw category rows and cells
        y_pos = 150
        for cat_idx, category in enumerate(categories):
            # Draw category label
            cv2.rectangle(screen, (50, y_pos), (200, y_pos + row_heights[cat_idx]), (220, 220, 220), -1)
            cv2.rectangle(screen, (50, y_pos), (200, y_pos + row_heights[cat_idx]), (0, 0, 0), 1)
            cv2.putText(screen, category, (60, y_pos + row_heights[cat_idx]//2 + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            self._add_region(f"category_{category.lower()}", 50, y_pos, 200, y_pos + row_heights[cat_idx], (220, 220, 220))
            
            # Draw cells for each product
            for prod_idx in range(3):
                x1 = 200 + prod_idx * col_width
                y1 = y_pos
                x2 = x1 + col_width
                y2 = y1 + row_heights[cat_idx]
                
                cv2.rectangle(screen, (x1, y1), (x2, y2), category_colors[cat_idx], -1)
                cv2.rectangle(screen, (x1, y1), (x2, y2), (0, 0, 0), 1)
                
                # Add content based on category
                content = ""
                if category == "Price":
                    prices = ["$99.99", "$199.99", "$349.99"]
                    content = prices[prod_idx]
                elif category == "Features":
                    features = ["Basic", "Standard", "Advanced"]
                    content = features[prod_idx]
                elif category == "Reviews":
                    reviews = ["★★★☆☆", "★★★★☆", "★★★★★"]
                    content = reviews[prod_idx]
                elif category == "Availability":
                    availability = ["In Stock", "2-3 Days", "Pre-order"]
                    content = availability[prod_idx]
                
                if content:
                    cv2.putText(screen, content, (x1 + 20, y1 + row_heights[cat_idx]//2 + 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                
                # For image category, draw a placeholder image
                if category == "Image":
                    image_margin = 20
                    cv2.rectangle(screen, 
                                 (x1 + image_margin, y1 + image_margin), 
                                 (x2 - image_margin, y2 - image_margin), 
                                 product_colors[prod_idx], -1)
                    cv2.putText(screen, f"Product {prod_idx+1}", 
                               (x1 + col_width//2 - 60, y1 + row_heights[cat_idx]//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                
                self._add_region(f"product_{prod_idx+1}_{category.lower()}", 
                                x1, y1, x2, y2, category_colors[cat_idx])
            
            y_pos += row_heights[cat_idx]
        
        return screen
    
    def _create_checkout_layout(self):
        """Create a checkout page layout with different functional areas."""
        screen = self._create_blank_screen()
        
        # Header
        cv2.rectangle(screen, (0, 0), (self.base_width, 100), (200, 200, 200), -1)
        cv2.putText(screen, "Checkout Page", (self.base_width//2 - 120, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
        self._add_region("header", 0, 0, self.base_width, 100, (200, 200, 200))
        
        # Left column - Cart summary
        cv2.rectangle(screen, (50, 150), (600, 750), (240, 240, 240), -1)
        cv2.rectangle(screen, (50, 150), (600, 750), (0, 0, 0), 1)
        cv2.putText(screen, "Cart Summary", (270, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        self._add_region("cart_summary", 50, 150, 600, 750, (240, 240, 240))
        
        # Cart items
        item_y = 230
        for i in range(3):
            cv2.rectangle(screen, (80, item_y), (570, item_y + 120), (255, 255, 255), -1)
            cv2.rectangle(screen, (80, item_y), (570, item_y + 120), (200, 200, 200), 1)
            
            # Item image placeholder
            cv2.rectangle(screen, (100, item_y + 10), (180, item_y + 100), (220, 220, 220), -1)
            
            # Item details
            cv2.putText(screen, f"Product {i+1}", (200, item_y + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(screen, f"Quantity: {i+1}", (200, item_y + 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)
            cv2.putText(screen, f"${(i+1)*99.99:.2f}", (450, item_y + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            self._add_region(f"cart_item_{i+1}", 80, item_y, 570, item_y + 120, (255, 255, 255))
            item_y += 140
        
        # Cart total
        cv2.rectangle(screen, (80, 650), (570, 720), (230, 240, 255), -1)
        cv2.rectangle(screen, (80, 650), (570, 720), (0, 0, 0), 1)
        cv2.putText(screen, "Total:", (100, 685), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(screen, "$599.97", (450, 685), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        self._add_region("cart_total", 80, 650, 570, 720, (230, 240, 255))
        
        # Right column - Customer Information
        cv2.rectangle(screen, (650, 150), (1200, 750), (240, 240, 240), -1)
        cv2.rectangle(screen, (650, 150), (1200, 750), (0, 0, 0), 1)
        cv2.putText(screen, "Customer Information", (820, 190), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        self._add_region("customer_info", 650, 150, 1200, 750, (240, 240, 240))
        
        # Form fields
        fields = ["Name", "Email", "Address", "City", "Zip Code", "Phone"]
        field_y = 230
        for field in fields:
            cv2.putText(screen, field + ":", (680, field_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.rectangle(screen, (680, field_y + 40), (1170, field_y + 80), (255, 255, 255), -1)
            cv2.rectangle(screen, (680, field_y + 40), (1170, field_y + 80), (200, 200, 200), 1)
            
            self._add_region(f"field_{field.lower().replace(' ', '_')}", 
                            680, field_y, 1170, field_y + 80, (255, 255, 255))
            field_y += 85
        
        # Right column - Payment Options
        cv2.rectangle(screen, (1250, 150), (1850, 750), (240, 240, 240), -1)
        cv2.rectangle(screen, (1250, 150), (1850, 750), (0, 0, 0), 1)
        cv2.putText(screen, "Payment Options", (1450, 190), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        self._add_region("payment_options", 1250, 150, 1850, 750, (240, 240, 240))
        
        # Payment methods
        payment_methods = ["Credit Card", "PayPal", "Apple Pay", "Google Pay"]
        method_colors = [(255, 220, 220), (220, 220, 255), (220, 255, 220), (255, 255, 220)]
        
        method_y = 230
        for i, method in enumerate(payment_methods):
            cv2.rectangle(screen, (1280, method_y), (1820, method_y + 80), method_colors[i], -1)
            cv2.rectangle(screen, (1280, method_y), (1820, method_y + 80), (200, 200, 200), 1)
            cv2.putText(screen, method, (1450, method_y + 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            
            self._add_region(f"payment_{method.lower().replace(' ', '_')}", 
                            1280, method_y, 1820, method_y + 80, method_colors[i])
            method_y += 100
        
        # Place Order button
        cv2.rectangle(screen, (1350, 650), (1750, 720), (0, 150, 0), -1)
        cv2.putText(screen, "PLACE ORDER", (1450, 695), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        self._add_region("place_order", 1350, 650, 1750, 720, (0, 150, 0))
        
        return screen
    
    def update_region_analytics(self, gaze_x, gaze_y):
        """Update analytics for the region containing the gaze point."""
        current_time = time.time()
        
        for region_name, region_data in self.regions.items():
            x1, y1, x2, y2 = region_data["coords"]
            
            # Check if gaze point is inside this region
            if x1 <= gaze_x <= x2 and y1 <= gaze_y <= y2:
                region_data["gaze_points"] += 1
                
                # If this is a new visit to the region, record the time
                if region_data["last_visit"] is None:
                    region_data["last_visit"] = current_time
                else:
                    # Add time spent in this region since last gaze point
                    region_data["total_time"] += current_time - region_data["last_visit"]
                    region_data["last_visit"] = current_time
                
                break  # Only one region can contain the gaze point
    
    def get_analytics_overlay(self):
        """Create a visualization of the current analytics."""
        overlay = self.current_screen.copy()
        
        # Sort regions by attention (gaze points)
        sorted_regions = sorted(
            self.regions.items(), 
            key=lambda x: x[1]["gaze_points"], 
            reverse=True
        )
        
        # Determine maximum values for normalization
        max_gaze = max([r[1]["gaze_points"] for r in sorted_regions]) if sorted_regions else 1
        max_time = max([r[1]["total_time"] for r in sorted_regions]) if sorted_regions else 1
        
        # Draw analytics on each region
        for region_name, region_data in sorted_regions:
            x1, y1, x2, y2 = region_data["coords"]
            gaze_points = region_data["gaze_points"]
            total_time = region_data["total_time"]
            
            # Skip regions with no data
            if gaze_points == 0:
                continue
            
            # Calculate opacity for heat overlay based on gaze points
            opacity = min(0.7, gaze_points / max_gaze * 0.7)
            
            # Create a semi-transparent overlay for the region
            # Color shifts from blue (cold) to red (hot) based on attention
            red = int(255 * (gaze_points / max_gaze))
            blue = int(255 * (1 - gaze_points / max_gaze))
            heat_color = (0, blue, red)
            
            # Create a colored overlay
            region_overlay = overlay[y1:y2, x1:x2].copy()
            color_layer = np.full(region_overlay.shape, heat_color, dtype=np.uint8)
            overlay[y1:y2, x1:x2] = cv2.addWeighted(region_overlay, 1 - opacity, color_layer, opacity, 0)
            
            # Add text with analytics
            cv2.putText(overlay, f"Views: {gaze_points}", (x1 + 10, y1 + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(overlay, f"Time: {total_time:.1f}s", (x1 + 10, y1 + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return overlay
    
    def save_analytics(self, timestamp=None):
        """Save the current analytics to a file."""
        if timestamp is None:
            timestamp = int(time.time())
        
        # Create a CSV file with the analytics
        csv_path = os.path.join(self.output_dir, f"layout_{self.current_layout}_analytics_{timestamp}.csv")
        with open(csv_path, "w") as f:
            f.write("Region,Gaze Points,Total Time (seconds)\n")
            
            # Sort regions by attention (gaze points)
            sorted_regions = sorted(
                self.regions.items(), 
                key=lambda x: x[1]["gaze_points"], 
                reverse=True
            )
            
            for region_name, region_data in sorted_regions:
                f.write(f"{region_name},{region_data['gaze_points']},{region_data['total_time']:.2f}\n")
        
        # Save an image with the analytics overlay
        img_path = os.path.join(self.output_dir, f"layout_{self.current_layout}_heatmap_{timestamp}.png")
        cv2.imwrite(img_path, self.get_analytics_overlay())
        
        print(f"Saved analytics to {csv_path} and {img_path}")
        return csv_path, img_path 